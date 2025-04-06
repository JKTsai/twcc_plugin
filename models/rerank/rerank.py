from typing import Optional

import requests
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.rerank_model import RerankModel


class TWCCRerankModel(RerankModel):
    """
    Reranker for TWCC API
    """

    API_URL = "https://api-ams.twcc.ai/api/models/rerank"

    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        """
        Invoke rerank model

        :param model: model name
        :param credentials: model credentials
        :param query: search query
        :param docs: docs for reranking
        :param score_threshold: score threshold
        :param top_n: top n
        :param user: unique user id
        :return: rerank result
        """
        if not docs:
            return RerankResult(model=model, docs=[])

        headers = {
            "X-API-KEY": credentials.get("twcc_api_key"),
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "query": query,
            "documents": docs,
            "parameters": {"top_n": top_n or len(docs)},
        }

        try:
            print(f"\n\n\nAPI key: \n\n {credentials.get("twcc_api_key")}\n\n\n\n")
            print(f"\n\n\nPayload: \n\n {payload}\n\n\n\n")
            response = requests.post(self.API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            rerank_documents = []
            for result in data.get("results", []):
                score = result["score"]
                index = result["index"]
                if 0 <= index < len(docs):
                    doc = docs[index]
                    rerank_doc = RerankDocument(index=index, text=doc, score=score)
                    if score_threshold is None or score >= score_threshold:
                        rerank_documents.append(rerank_doc)

            return RerankResult(model=model, docs=rerank_documents)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise InvokeAuthorizationError(str(e))
            elif e.response.status_code == 400:
                raise InvokeBadRequestError(str(e))
            elif e.response.status_code == 429:
                raise InvokeRateLimitError(str(e))
            elif e.response.status_code >= 500:
                raise InvokeServerUnavailableError(str(e))
            else:
                raise InvokeError(str(e))
        except requests.exceptions.ConnectionError as e:
            raise InvokeConnectionError(str(e))
        except Exception as e:
            raise InvokeError(str(e))

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [requests.exceptions.ConnectionError],
            InvokeServerUnavailableError: [requests.exceptions.HTTPError],
            InvokeRateLimitError: [requests.exceptions.HTTPError],
            InvokeAuthorizationError: [requests.exceptions.HTTPError],
            InvokeBadRequestError: [requests.exceptions.HTTPError],
        }

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self.invoke(
                model=model,
                credentials=credentials,
                query="What is the capital of the United States?",
                docs=[
                    "Carson City is the capital city of the American state of Nevada.",
                    "The capital of the United States is Washington, D.C.",
                ],
                score_threshold=0.5,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))
