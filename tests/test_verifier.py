"""Offline checks; these do not measure model classification accuracy."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
import hw2


def response(status='valid', *, profile=True, blocks=False, **changes):
    verdict = dict(verdict=status, person_id=42,
                   checked_fields=['name', 'city', 'jobs', 'education', 'skills'],
                   discrepancies=[])
    verdict.update(changes)
    messages = []
    if profile:
        evidence = json.dumps(dict(id=42, name='Synthetic Candidate', city='City',
                                   experience=[], education=[], skills=[]))
        messages.append(ToolMessage(
            content=[{'type': 'text', 'text': evidence}] if blocks else evidence,
            name='get_linkedin_profile', tool_call_id='test'))
    messages.append(AIMessage(content=json.dumps(verdict)))
    return {'messages': messages}


class VerdictTests(unittest.TestCase):
    def test_real_tool_evidence_and_block_content(self):
        self.assertEqual(hw2._verified_score(response(blocks=True)), 0.95)

    def test_explanation_before_json(self):
        result = response()
        result['messages'][-1].content = 'All fields checked.\n' + result['messages'][-1].content
        self.assertEqual(hw2._verified_score(result), 0.95)

    def test_ambiguous_verdicts_rejected(self):
        result = response()
        result['messages'][-1].content *= 2
        with self.assertRaises(ValueError):
            hw2._verified_score(result)

    def test_fabricated_profile_is_not_evidence(self):
        with self.assertRaises(ValueError):
            hw2._verified_score(response(profile=False))

    def test_wrong_identity_is_not_evidence(self):
        with self.assertRaises(ValueError):
            hw2._verified_score(response(person_id=99))

    def test_incomplete_coverage_cannot_pass(self):
        with self.assertRaises(ValueError):
            hw2._verified_score(response(checked_fields=['name']))

    def test_contradiction(self):
        result = response('discrepancy', discrepancies=[
            dict(field='education.degree', claim='MSc', evidence='BSc')])
        self.assertEqual(hw2._verified_score(result), 0.05)

    def test_uncertainty(self):
        self.assertEqual(hw2._verified_score(response('uncertain', profile=False)), 0.5)

    def test_agent_factory(self):
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'offline-test-only'}):
            self.assertTrue(hasattr(hw2.build_agent([]), 'ainvoke'))


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_limit_and_independent_messages(self):
        state = SimpleNamespace(active=0, peak=0, prompts=[])

        class Agent:
            async def ainvoke(self, payload, config):
                state.active += 1
                state.peak = max(state.peak, state.active)
                state.prompts.append(payload)
                await asyncio.sleep(0.01)
                state.active -= 1
                return response()

        cvs = {f'input-{i}.pdf': f'CV text {i}' for i in range(8)}
        scores = await hw2.score_cvs(Agent(), cvs)
        self.assertEqual(scores, dict.fromkeys(cvs, 0.95))
        self.assertLessEqual(state.peak, 3)
        self.assertGreater(state.peak, 1)
        for payload in state.prompts:
            self.assertEqual(len(payload['messages']), 1)
            self.assertEqual(set(json.loads(payload['messages'][0]['content'])), {'cv_text'})

    async def test_one_failure_does_not_abort_batch(self):
        attempts = {}

        class Agent:
            async def ainvoke(self, payload, config):
                text = json.loads(payload['messages'][0]['content'])['cv_text']
                attempts[text] = attempts.get(text, 0) + 1
                if text == 'bad':
                    raise ConnectionError('offline failure')
                if text == 'retry' and attempts[text] == 1:
                    return {'messages': [AIMessage(content='not JSON')]}
                return response()

        with patch('hw2.asyncio.sleep', return_value=None):
            scores = await hw2.score_cvs(Agent(), {
                'a.pdf': 'good', 'b.pdf': 'bad', 'c.pdf': 'retry', 'd.pdf': ''})
        self.assertEqual(scores, {'a.pdf': 0.95, 'b.pdf': 0.5,
                                  'c.pdf': 0.95, 'd.pdf': 0.5})
        self.assertEqual(attempts['bad'], 2)
        self.assertEqual(attempts['retry'], 2)

    async def test_timeout_is_bounded(self):
        class Agent:
            async def ainvoke(self, *args, **kwargs):
                await asyncio.sleep(60)

        real_wait_for = asyncio.wait_for

        async def short_wait(awaitable, timeout):
            self.assertEqual(timeout, 100)
            return await real_wait_for(awaitable, timeout=0.001)

        with patch('hw2.asyncio.wait_for', side_effect=short_wait):
            self.assertEqual(await hw2.score_cvs(Agent(), {'x.pdf': 'text'}),
                             {'x.pdf': 0.5})


if __name__ == '__main__':
    unittest.main()
