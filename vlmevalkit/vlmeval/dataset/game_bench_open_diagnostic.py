import json
import os.path as osp
import re
import string
from collections import Counter

import pandas as pd

from .image_mixedqa import (
    ImageMixedQADataset,
    build_mixedqa_choices,
    split_mixedqa_paths,
)
from ..smp import dump, get_intermediate_file_path, load, toliststr


_CHOICE_REFERENCE = re.compile(r'^\s*[A-H](?:\s*[,;/|&+]\s*[A-H])*\s*[.!]?\s*$', re.I)
_FINAL_ANSWER_MARKER = re.compile(
    r'(?:^|\n)\s*(?:\d+\s*[.)]\s*)?(?:\*{1,2}|_{1,2})?'
    r'final\s+answer\s*[:\uff1a]?\s*(?:\*{1,2}|_{1,2})?[ \t]*',
    re.I,
)
_FIELD_LINE = re.compile(r'^\s*([A-Za-z][A-Za-z /-]*?)\s*:\s*(.*?)\s*$', re.M)
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")


def canonical_diagnostic_field(label):
    return re.sub(r'[^a-z0-9]+', '_', str(label).strip().lower()).strip('_')


def normalize_diagnostic_evidence(value):
    value = str(value).lower()
    value = re.sub(r'\s+', ' ', value).strip()
    value = re.sub(r'\s*([,;/|&=:])\s*', r'\1', value)
    return value.rstrip(' .\u3002')


def diagnostic_analysis_text(prediction):
    """Return only the evidence section; a long exact final answer is exempt from the word budget."""
    text = str(prediction).replace('\r\n', '\n').replace('\r', '\n')
    markers = list(_FINAL_ANSWER_MARKER.finditer(text))
    return text[:markers[-1].start()] if markers else text


def _strip_answer_wrappers(answer):
    answer = str(answer).strip()
    answer = re.sub(r'^```(?:json|text)?\s*', '', answer, flags=re.I)
    answer = re.sub(r'\s*```$', '', answer)
    answer = re.sub(r'^\s*(?:[-*]\s+|\d+\s*[.)]\s*)', '', answer)
    answer = re.sub(r'^\s*(?:answer|option)\s*(?:is|:)?\s*', '', answer, flags=re.I)

    boxed = re.fullmatch(r'\\boxed\s*\{(.*)\}', answer, flags=re.S)
    if boxed:
        answer = boxed.group(1).strip()
    tagged = re.fullmatch(r'<answer>\s*(.*?)\s*</answer>', answer, flags=re.I | re.S)
    if tagged:
        answer = tagged.group(1).strip()

    answer = re.sub(r'^(?:\*{1,2}|_{1,2})(.*?)(?:\*{1,2}|_{1,2})$', r'\1', answer, flags=re.S)
    return answer.strip()


def extract_diagnostic_final_answer(prediction):
    """Extract the answer section while retaining the full prediction elsewhere."""
    text = str(prediction).replace('\r\n', '\n').replace('\r', '\n').strip()
    if not text:
        return ''

    markers = list(_FINAL_ANSWER_MARKER.finditer(text))
    if markers:
        tail = text[markers[-1].end():].strip()
        if not tail:
            return ''
        # The benchmark requests a separate final-answer line. Keeping one line
        # prevents accidental trailing commentary from entering exact matching.
        candidate = next(
            (line.strip() for line in tail.split('\n') if line.strip() and line.strip() != '```'),
            '',
        )
    else:
        candidate = next((line.strip() for line in reversed(text.split('\n')) if line.strip()), '')

    return _strip_answer_wrappers(candidate)


def normalize_diagnostic_choice(answer):
    """Return a canonical option set, or an empty string for invalid output."""
    answer = _strip_answer_wrappers(answer)
    letters = re.findall(r'(?<![A-Z])([A-H])(?![A-Z])', answer.upper())
    if not letters or len(letters) != len(set(letters)):
        return ''
    return ''.join(sorted(letters))


def normalize_diagnostic_open_answer(answer):
    """Ignore presentation-only differences without weakening structured answers."""
    answer = _strip_answer_wrappers(answer).lower()
    answer = re.sub(r'\s+', ' ', answer).strip()
    answer = re.sub(r'\s*([,;|&=:\[\]\(\)])\s*', r'\1', answer)
    return answer.rstrip(' .\u3002')


def score_diagnostic_answer(prediction, reference):
    extracted = extract_diagnostic_final_answer(prediction)
    reference = str(reference).strip()
    if _CHOICE_REFERENCE.fullmatch(reference):
        normalized_prediction = normalize_diagnostic_choice(extracted)
        normalized_reference = normalize_diagnostic_choice(reference)
        hit = int(bool(normalized_prediction) and normalized_prediction == normalized_reference)
    else:
        normalized_prediction = normalize_diagnostic_open_answer(extracted)
        normalized_reference = normalize_diagnostic_open_answer(reference)
        hit = int(bool(normalized_prediction) and normalized_prediction == normalized_reference)
    return extracted, hit, normalized_prediction, normalized_reference


class GameBenchOpenDiagnosticDataset(ImageMixedQADataset):
    """Open-response Game-Bench diagnostic set with deterministic final-answer scoring."""

    TYPE = 'MixQA'
    DATASET_URL = {
        'Game_Bench_Open_Diagnostic_72_major_template_coverage': (
            '/root/LMUData/VLMEvalKit/resource/Game-Bench-Scaling-1200-official/'
            'Game_Bench_Open_Diagnostic_72_major_template_coverage.tsv'
        ),
    }
    DATASET_MD5 = {
        'Game_Bench_Open_Diagnostic_72_major_template_coverage': None,
    }
    ANNOTATION_FILENAME = 'Game_Bench_Open_Diagnostic_72_evidence_annotations.json'

    def load_data(self, dataset):
        data = super().load_data(dataset)

        # In this TSV `image` is legacy path metadata, not base64 image data.
        # Removing it before ImageBaseDataset.__init__ prevents base64 decoding.
        if 'image' in data:
            data = data.drop(columns=['image'])

        required = ['index', 'question', 'image_path']
        missing = [column for column in required if column not in data]
        if missing:
            raise ValueError(f'{dataset}: missing required columns: {missing}')
        if data['index'].isna().any() or not data['index'].is_unique:
            raise ValueError(f'{dataset}: `index` must be non-null and unique.')
        if data['question'].isna().any():
            raise ValueError(f'{dataset}: `question` contains null values.')

        for index, value in zip(data['index'], data['image_path']):
            try:
                paths = [path.strip() for path in split_mixedqa_paths(value) if str(path).strip()]
            except Exception as err:
                raise ValueError(f'{dataset}: malformed image_path at index {index}: {value!r}') from err
            if not paths:
                raise ValueError(f'{dataset}: empty image_path at index {index}.')

        old_answer = data['answer'] if 'answer' in data else pd.Series(pd.NA, index=data.index)
        if 'gold_final_answer' in data:
            answer = data['gold_final_answer'].where(data['gold_final_answer'].notna(), old_answer)
        else:
            answer = old_answer
        if answer.isna().any():
            bad = data.loc[answer.isna(), 'index'].tolist()
            raise ValueError(f'{dataset}: missing final-answer ground truth at indexes {bad}.')
        data['answer'] = [str(value).strip() for value in answer]
        return data

    def post_build(self, dataset):
        self.evidence_annotations = {}
        annotation_path = osp.join(self.data_dir, self.ANNOTATION_FILENAME)
        if not osp.exists(annotation_path):
            return
        annotations = load(annotation_path)
        if not isinstance(annotations, dict):
            raise ValueError(f'{annotation_path}: annotation file must contain a JSON object.')
        records = annotations.get('records', {})
        if not isinstance(records, dict):
            raise ValueError(f'{annotation_path}: `records` must be a JSON object keyed by dataset index.')
        self.evidence_annotations = {str(index): value for index, value in records.items() if isinstance(value, dict)}

    @staticmethod
    def capability_family(line):
        dimensions = str(line.get('manual_review_dimensions', ''))
        if '三维表征' in dimensions:
            return 'spatial_3d'
        if '规则覆盖' in dimensions:
            return 'constraint_reasoning'
        if '每步状态' in dimensions:
            return 'state_transition'
        if '路径合法' in dimensions:
            return 'path_planning'
        return 'perception'

    def required_diagnostic_fields(self, line, choices):
        # These labels only structure the record. They deliberately introduce
        # no task-specific rule, solving procedure, or candidate heuristic.
        return ('Observations', 'Reasoning summary')

    def diagnostic_protocol(self, line, choices):
        # Keep the TSV question verbatim. This neutral logging format makes
        # responses comparable without teaching the model how to solve a game.
        return (
            'Response format for diagnostic logging:\n'
            'Observations: <task-relevant visual or state evidence>\n'
            'Reasoning summary: <your reasoning>\n'
            'Final answer: <ANSWER>'
        )

    @staticmethod
    def diagnostic_image_instruction(line, image_paths, choices):
        if len(image_paths) <= 1:
            return ''

        mode = str(line.get('diagnostic_mode', '')).strip().lower()
        if mode == 'visual_or_panel_candidates' and choices and len(image_paths) == len(choices) + 1:
            lines = ['Image 1 is the main question image.']
            for image_number, option in enumerate(choices, start=2):
                lines.append(f'Image {image_number} is option {option}.')
            lines.append('Refer to candidate images by their option letters, not their image numbers.')
            return '\n'.join(lines) + '\n'

        return (
            'Images are provided in their stored order. Some images may contain multiple labeled panels; '
            'use labels visible inside the images and do not assume that one image equals one answer option.\n'
        )

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_paths = toliststr(line['image_path'])
        choices = build_mixedqa_choices(line)

        prompt = str(line['question']).strip() + '\n'
        hint = line.get('hint', None)
        if hint is not None and not pd.isna(hint) and str(hint).strip():
            prompt = f'Hint: {hint}\n' + prompt
        if choices:
            prompt += 'Options:\n'
            prompt += '\n'.join(f'{key}. {value}' for key, value in choices.items()) + '\n'
        prompt += self.diagnostic_image_instruction(line, image_paths, choices)
        prompt += '\n' + self.diagnostic_protocol(line, choices) + '\n'

        messages = [dict(type='image', value=path) for path in image_paths]
        messages.append(dict(type='text', value=prompt))
        return messages

    @staticmethod
    def parse_diagnostic_fields(prediction):
        fields = {}
        for label, value in _FIELD_LINE.findall(str(prediction)):
            canonical = canonical_diagnostic_field(label)
            if canonical and canonical not in fields:
                fields[canonical] = value.strip()
        return fields

    @staticmethod
    def has_repeated_diagnostic_line(prediction):
        normalized_lines = []
        for line in diagnostic_analysis_text(prediction).splitlines():
            normalized = normalize_diagnostic_evidence(line)
            if normalized:
                normalized_lines.append(normalized)
        return any(count > 1 for count in Counter(normalized_lines).values())

    def diagnostic_record(self, line, prediction):
        choices = build_mixedqa_choices(line)
        required_labels = self.required_diagnostic_fields(line, choices)
        required_fields = [canonical_diagnostic_field(label) for label in required_labels]
        parsed_fields = self.parse_diagnostic_fields(prediction)
        missing_fields = [field for field in required_fields if not parsed_fields.get(field, '').strip()]
        extracted_answer = extract_diagnostic_final_answer(prediction)
        evidence_word_count = len(_WORD_PATTERN.findall(diagnostic_analysis_text(prediction)))
        total_word_count = len(_WORD_PATTERN.findall(str(prediction)))
        has_final_marker = bool(_FINAL_ANSWER_MARKER.search(str(prediction)))
        repetition_detected = self.has_repeated_diagnostic_line(prediction)

        annotation = self.evidence_annotations.get(str(line['index']), {})
        expected_fields = annotation.get('expected_fields', {}) if annotation else {}
        evidence_total = 0
        evidence_matched = 0
        for field, expected in expected_fields.items():
            field_name = canonical_diagnostic_field(field)
            model_value = parsed_fields.get(field_name, '')
            accepted_values = expected if isinstance(expected, list) else [expected]
            if model_value:
                evidence_total += 1
                if any(
                    normalize_diagnostic_evidence(model_value) == normalize_diagnostic_evidence(value)
                    for value in accepted_values
                ):
                    evidence_matched += 1

        return {
            'diagnostic_family': self.capability_family(line),
            'diagnostic_word_count': evidence_word_count,
            'diagnostic_total_word_count': total_word_count,
            'has_final_answer_marker': int(has_final_marker),
            'required_fields': json.dumps(required_fields),
            'parsed_fields': json.dumps(parsed_fields, ensure_ascii=False),
            'missing_required_fields': json.dumps(missing_fields),
            'repetition_detected': int(repetition_detected),
            'format_valid': int(
                not missing_fields
                and bool(extracted_answer)
                and has_final_marker
            ),
            'evidence_annotation_available': int(bool(expected_fields)),
            'evidence_fields_total': evidence_total,
            'evidence_fields_matched': evidence_matched,
            'evidence_match_rate': evidence_matched / evidence_total if evidence_total else None,
        }

    @staticmethod
    def diagnostic_report(data):
        reports = []
        groupings = [('overall', None), ('category', 'category'), ('family', 'diagnostic_family'), ('subfield', 'subfield')]
        metric_columns = [
            'hit', 'format_valid', 'has_final_answer_marker', 'repetition_detected',
        ]
        for group_name, column in groupings:
            groups = [('Overall', data)] if column is None or column not in data else data.groupby(column, dropna=False)
            for label, subset in groups:
                row = {'group_type': group_name, 'group': str(label), 'samples': len(subset)}
                for metric in metric_columns:
                    row[metric] = float(subset[metric].mean())
                annotated = subset[subset['evidence_annotation_available'] == 1]
                row['annotated_samples'] = len(annotated)
                row['evidence_match_rate'] = (
                    float(annotated['evidence_match_rate'].dropna().mean())
                    if len(annotated) and annotated['evidence_match_rate'].notna().any() else None
                )
                reports.append(row)
        return pd.DataFrame(reports)

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        for key in list(data.keys()):
            normalized_key = key if key in string.ascii_uppercase else key.lower()
            if normalized_key != key:
                data[normalized_key] = data.pop(key)
        if 'prediction' not in data:
            raise ValueError(f'{eval_file}: missing `prediction` column.')

        metadata = self.data
        answer_map = {str(index): answer for index, answer in zip(metadata['index'], metadata['answer'])}
        data['_diagnostic_index'] = [str(index) for index in data['index']]
        data = data[data['_diagnostic_index'].isin(answer_map)].copy()
        if not len(data):
            raise ValueError(f'{eval_file}: no prediction indexes match the diagnostic dataset.')

        extracted_answers = []
        hits = []
        logs = []
        diagnostic_records = []
        for _, item in data.iterrows():
            reference = answer_map[item['_diagnostic_index']]
            extracted, hit, normalized_prediction, normalized_reference = score_diagnostic_answer(
                item['prediction'], reference
            )
            extracted_answers.append(extracted)
            hits.append(hit)
            logs.append(
                f'Extracted: {extracted!r}; normalized prediction: {normalized_prediction!r}; '
                f'normalized reference: {normalized_reference!r}.'
            )
            diagnostic_records.append(self.diagnostic_record(item, item['prediction']))

        data['extracted_answer'] = extracted_answers
        data['hit'] = hits
        data['log'] = logs
        for field in diagnostic_records[0]:
            data[field] = [record[field] for record in diagnostic_records]
        data = data.drop(columns=['_diagnostic_index'])

        result_file = get_intermediate_file_path(eval_file, '_diagnostic_result')
        dump(data, result_file)

        from .utils.multiple_choice import report_acc

        accuracy = report_acc(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(accuracy, score_file)
        diagnostic_report_file = get_intermediate_file_path(eval_file, '_diagnostic_report', 'csv')
        dump(self.diagnostic_report(data), diagnostic_report_file)
        return accuracy
