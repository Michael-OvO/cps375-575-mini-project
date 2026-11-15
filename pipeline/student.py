"""Sample from a Tinker-hosted model through its recommended renderer."""

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


class Sampler:
    def __init__(self, model: str, checkpoint_path: str | None = None):
        tokenizer = get_tokenizer(model)
        self.renderer = renderers.get_renderer(
            model_info.get_recommended_renderer_name(model), tokenizer
        )
        service = tinker.ServiceClient()
        if checkpoint_path:
            self.client = service.create_sampling_client(model_path=checkpoint_path).result()
        else:
            self.client = service.create_sampling_client(base_model=model).result()

    async def answer(
        self, text: str, max_tokens: int, temperature: float = 0.0, n: int = 1
    ) -> list[str]:
        # Stripped like the graded inspect adapter, so dev evals see the same prompt tokens.
        prompt = self.renderer.build_generation_prompt([{"role": "user", "content": text.strip()}])
        params = tinker.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=self.renderer.get_stop_sequences(),
            top_p=1.0,
        )
        result = await self.client.sample_async(prompt=prompt, sampling_params=params, num_samples=n)
        return [_text(self.renderer.parse_response(seq.tokens)[0]["content"]) for seq in result.sequences]
