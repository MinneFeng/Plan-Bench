import os.path as osp
import re
import string

import pandas as pd

from .image_mixedqa import (
    ImageMixedQADataset,
    build_mixedqa_choices,
    normalize_mixedqa_choice_answer,
    normalize_mixedqa_open_answer,
)
from ..smp import dump, get_intermediate_file_path, load, toliststr
from ..utils import can_infer


_FINAL_ANSWER_MARKER = re.compile(
    r'(?:^|\n)\s*(?:\d+\s*[.)]\s*)?(?:\*{1,2}|_{1,2})?'
    r'final\s+answer\s*[:\uff1a]?\s*(?:\*{1,2}|_{1,2})?[ \t]*',
    re.I,
)


def extract_visible_rationale_answer(prediction):
    """Return the answer following the final explicit `Final answer` marker."""
    text = str(prediction).replace('\r\n', '\n').replace('\r', '\n').strip()
    markers = list(_FINAL_ANSWER_MARKER.finditer(text))
    if not markers:
        return text

    tail = text[markers[-1].end():].strip()
    if not tail:
        return ''
    answer = next((line.strip() for line in tail.split('\n') if line.strip()), '')
    answer = re.sub(r'^\\boxed\s*\{(.*)\}$', r'\1', answer).strip()
    answer = re.sub(r'^<answer>\s*(.*?)\s*</answer>$', r'\1', answer, flags=re.I).strip()
    answer = re.sub(r'^(?:\*{1,2}|_{1,2})(.*?)(?:\*{1,2}|_{1,2})$', r'\1', answer).strip()
    return answer


class GameBenchVisibleRationaleDataset(ImageMixedQADataset):
    """Original 80-row MixedQA set with an explicitly visible rationale format."""

    TYPE = 'MixQA'
    DATASET_URL = {
        'Game_Bench_Scaling_1200_controlled_scaling_sanity_80_visible_rationale': (
            '/root/LMUData/VLMEvalKit/resource/Game-Bench-Scaling-1200-official/'
            'Game_Bench_Scaling_1200_controlled_scaling_sanity_80_clean.tsv'
        ),
    }
    DATASET_MD5 = {
        'Game_Bench_Scaling_1200_controlled_scaling_sanity_80_visible_rationale': None,
    }

    @staticmethod
    def image_reference_instruction(image_paths, choices):
        if len(image_paths) <= 1:
            return ''
        lines = ['Image 1 is the main question image.']
        for image_number, option in enumerate(choices, start=2):
            if image_number > len(image_paths):
                break
            lines.append(f'Image {image_number} is option {option}.')
        if choices:
            lines.append('Refer to candidate images by option letter, not image number.')
        return '\n'.join(lines) + '\n'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_paths = toliststr(line['image_path'])
        choices = build_mixedqa_choices(line)
        prompt = ''
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {line["question"]}\n'
        if choices:
            prompt += 'Options:\n'
            prompt += '\n'.join(f'{key}. {value}' for key, value in choices.items()) + '\n'
        prompt += self.image_reference_instruction(image_paths, choices)
        prompt += (
            'Give a concise visible rationale using exactly this format:\n'
            'Observations: <task-relevant visual or state evidence>\n'
            'Reasoning: <how the stated rules support the answer>\n'
            'Final answer: <required answer format>\n'
        )

        messages = [dict(type='image', value=path) for path in image_paths]
        messages.append(dict(type='text', value=prompt))
        return messages

    def eval_item(self, item):
        choices = build_mixedqa_choices(item)
        question_type = self.infer_question_type(item, has_options=bool(choices))
        extracted = extract_visible_rationale_answer(item['prediction'])

        if question_type == 'open':
            prediction = normalize_mixedqa_open_answer(extracted)
            answer = normalize_mixedqa_open_answer(item['GT'])
            hit = int(bool(prediction) and prediction == answer)
        elif question_type in ['multiple_choice', 'single_or_multiple_choice']:
            prediction = normalize_mixedqa_choice_answer(extracted, choices)
            answer = normalize_mixedqa_choice_answer(item['GT'], choices)
            hit = int(bool(prediction) and prediction == answer)
        else:
            prediction = can_infer(extracted, choices)
            answer = item['GT']
            hit = int(prediction == answer)

        return dict(
            extracted_answer=extracted,
            hit=hit,
            log=f'Extracted answer: {extracted!r}; normalized prediction: {prediction!r}; answer: {answer!r}.',
        )

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        data['prediction'] = [str(value) for value in data['prediction']]
        for key in list(data.keys()):
            normalized_key = key if key in string.ascii_uppercase else key.lower()
            if normalized_key != key:
                data[normalized_key] = data.pop(key)

        answer_map = {str(index): answer for index, answer in zip(self.data['index'], self.data['answer'])}
        data['_visible_rationale_index'] = [str(index) for index in data['index']]
        data = data[data['_visible_rationale_index'].isin(answer_map)].copy()
        if not len(data):
            raise ValueError(f'{eval_file}: no prediction indexes match the visible-rationale dataset.')
        data['GT'] = [answer_map[index] for index in data['_visible_rationale_index']]

        results = [self.eval_item(item) for _, item in data.iterrows()]
        data['extracted_answer'] = [result['extracted_answer'] for result in results]
        data['hit'] = [result['hit'] for result in results]
        data['log'] = [result['log'] for result in results]
        data = data.drop(columns=['GT', '_visible_rationale_index'])

        result_file = get_intermediate_file_path(eval_file, '_visible_rationale_result')
        dump(data, result_file)
        from .utils.multiple_choice import report_acc

        accuracy = report_acc(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(accuracy, score_file)
        return accuracy
