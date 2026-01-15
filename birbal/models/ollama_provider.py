from ollama import Client
from birbal.models.base import AIProvider
from birbal.config import config


class OllamaEmbeddings:
    model: str
    """Model name to use."""

    dimensions: int | None = None
    """Dimensions of generated vector embedding"""

    mirostat: int | None = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: `0`, `0` = disabled, `1` = Mirostat, `2` = Mirostat 2.0)"""

    mirostat_eta: float | None = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: `0.1`)"""

    mirostat_tau: float | None = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: `5.0`)"""

    num_ctx: int | None = None
    """Sets the size of the context window used to generate the
    next token. (Default: `2048`)	"""

    num_gpu: int | None = None
    """The number of GPUs to use. On macOS it defaults to `1` to
    enable metal support, `0` to disable."""

    keep_alive: int | None = None
    """Controls how long the model will stay loaded into memory
    following the request (default: `5m`)
    """

    num_thread: int | None = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    repeat_last_n: int | None = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: `64`, `0` = disabled, `-1` = `num_ctx`)"""

    repeat_penalty: float | None = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., `1.5`)
    will penalize repetitions more strongly, while a lower value (e.g., `0.9`)
    will be more lenient. (Default: `1.1`)"""

    temperature: float | None = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: `0.8`)"""

    stop: list[str] | None = None
    """Sets the stop tokens to use."""

    tfs_z: float | None = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., `2.0`) will reduce the
    impact more, while a value of `1.0` disables this setting. (default: `1`)"""

    top_k: int | None = None
    """Reduces the probability of generating nonsense. A higher value (e.g. `100`)
    will give more diverse answers, while a lower value (e.g. `10`)
    will be more conservative. (Default: `40`)"""

    top_p: float | None = None
    """Works together with top-k. A higher value (e.g., `0.95`) will lead
    to more diverse text, while a lower value (e.g., `0.5`) will
    generate more focused and conservative text. (Default: `0.9`)"""

    def __init__(self, host, model, dimensions, **kwargs):
        self.client = Client(host=host)
        self.model = model
        self.dimensions = dimensions
        self.options = kwargs

    def embed_documents(self, texts):
        if not self.client:
            msg = (
                "Ollama client is not initialized. "
                "Please ensure Ollama is running and the model is loaded."
            )
            raise ValueError(msg)

        response = self.client.embed(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            options=self.options,
        )

        return response["embeddings"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class OllamaLLM:
    model: str
    """Model name to use."""

    reasoning: bool | str | None = None
    """Controls the reasoning/thinking mode for [supported models](https://ollama.com/search?c=thinking).

    - `True`: Enables reasoning mode. The model's reasoning process will be
        captured and returned separately in the `additional_kwargs` of the
        response message, under `reasoning_content`. The main response
        content will not include the reasoning tags.
    - `False`: Disables reasoning mode. The model will not perform any reasoning,
        and the response will not include any reasoning content.
    - `None` (Default): The model will use its default reasoning behavior. Note
        however, if the model's default behavior *is* to perform reasoning, think tags
        (`<think>` and `</think>`) will be present within the main response content
        unless you set `reasoning` to `True`.
    - `str`: e.g. `'low'`, `'medium'`, `'high'`. Enables reasoning with a custom
        intensity level. Currently, this is only supported `gpt-oss`. See the
        [Ollama docs](https://github.com/ollama/ollama-python/blob/da79e987f0ac0a4986bf396f043b36ef840370bc/ollama/_types.py#L210)
        for more information.
    """

    validate_model_on_init: bool = False
    """Whether to validate the model exists in Ollama locally on initialization.

    !!! version-added "Added in `langchain-ollama` 0.3.4"
    """

    mirostat: int | None = None
    """Enable Mirostat sampling for controlling perplexity.

    (Default: `0`, `0` = disabled, `1` = Mirostat, `2` = Mirostat 2.0)
    """

    mirostat_eta: float | None = None
    """Influences how quickly the algorithm responds to feedback from generated text.

    A lower learning rate will result in slower adjustments, while a higher learning
    rate will make the algorithm more responsive.

    (Default: `0.1`)
    """

    mirostat_tau: float | None = None
    """Controls the balance between coherence and diversity of the output.

    A lower value will result in more focused and coherent text.

    (Default: `5.0`)
    """

    num_ctx: int | None = None
    """Sets the size of the context window used to generate the next token.

    (Default: `2048`)
    """

    num_gpu: int | None = None
    """The number of GPUs to use.

    On macOS it defaults to `1` to enable metal support, `0` to disable.
    """

    num_thread: int | None = None
    """Sets the number of threads to use during computation.

    By default, Ollama will detect this for optimal performance. It is recommended to
    set this value to the number of physical CPU cores your system has (as opposed to
    the logical number of cores).
    """

    num_predict: int | None = None
    """Maximum number of tokens to predict when generating text.

    (Default: `128`, `-1` = infinite generation, `-2` = fill context)
    """

    repeat_last_n: int | None = None
    """Sets how far back for the model to look back to prevent repetition.

    (Default: `64`, `0` = disabled, `-1` = `num_ctx`)
    """

    repeat_penalty: float | None = None
    """Sets how strongly to penalize repetitions.

    A higher value (e.g., `1.5`) will penalize repetitions more strongly, while a
    lower value (e.g., `0.9`) will be more lenient. (Default: `1.1`)
    """

    temperature: float | None = None
    """The temperature of the model.

    Increasing the temperature will make the model answer more creatively.

    (Default: `0.8`)
    """

    seed: int | None = None
    """Sets the random number seed to use for generation.

    Setting this to a specific number will make the model generate the same text for the
    same prompt.
    """

    stop: list[str] | None = None
    """Sets the stop tokens to use."""

    tfs_z: float | None = None
    """Tail free sampling.

    Used to reduce the impact of less probable tokens from the output.

    A higher value (e.g., `2.0`) will reduce the impact more, while a value of `1.0`
    disables this setting.

    (Default: `1`)
    """

    top_k: int | None = None
    """Reduces the probability of generating nonsense.

    A higher value (e.g. `100`) will give more diverse answers, while a lower value
    (e.g. `10`) will be more conservative.

    (Default: `40`)
    """

    top_p: float | None = None
    """Works together with top-k.

    A higher value (e.g., `0.95`) will lead to more diverse text, while a lower value
    (e.g., `0.5`) will generate more focused and conservative text.

    (Default: `0.9`)
    """

    keep_alive: int | str | None = None
    """How long the model will stay loaded into memory."""

    base_url: str | None = None
    """Base url the model is hosted under.

    If none, defaults to the Ollama client default.

    Supports `userinfo` auth in the format `http://username:password@localhost:11434`.
    Useful if your Ollama server is behind a proxy.

    !!! warning
        `userinfo` is not secure and should only be used for local testing or
        in secure environments. Avoid using it in production or over unsecured
        networks.

    !!! note
        If using `userinfo`, ensure that the Ollama server is configured to
        accept and validate these credentials.

    !!! note
        `userinfo` headers are passed to both sync and async clients.

    """

    client_kwargs: dict | None = {}
    """Additional kwargs to pass to the httpx clients. Pass headers in here.

    These arguments are passed to both synchronous and async clients.

    Use `sync_client_kwargs` and `async_client_kwargs` to pass different arguments
    to synchronous and asynchronous clients.
    """

    async_client_kwargs: dict | None = {}
    """Additional kwargs to merge with `client_kwargs` before passing to httpx client.

    These are clients unique to the async client; for shared args use `client_kwargs`.

    For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#asyncclient).
    """

    sync_client_kwargs: dict | None = {}
    """Additional kwargs to merge with `client_kwargs` before passing to httpx client.

    These are clients unique to the sync client; for shared args use `client_kwargs`.

    For a full list of the params, see the [httpx documentation](https://www.python-httpx.org/api/#client).
    """

    def __init__(self, host, model, stream, **kwargs):
        self.client = Client(host=host)
        self.model = model
        self.stream = stream
        self.options = kwargs

    def invoke(self, messages):
        """
        Note: Native Ollama uses [{'role': 'user', 'content': '...'}] format.
        """
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=self.stream,
            options=self.options,
        )

        if self.stream:
            return response

        return response["message"]["content"]


class OllamaProvider(AIProvider):
    def get_embedder(self):
        return OllamaEmbeddings(
            host=config["embed_host"],
            model=config["embed_model"],
            dimensions=config["vector_dims"],
        )

    def get_llm(self):
        return OllamaLLM(
            host=config["llm_host"],
            model=config["large_language_model"],
            stream=True,
            num_ctx=config["context_window_size"],
            temperature=0,
        )
