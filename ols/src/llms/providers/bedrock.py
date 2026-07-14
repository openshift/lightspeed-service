"""AWS Bedrock provider implementation."""

import logging
from typing import Any, Optional
from urllib.parse import urlparse

import boto3
import httpx
from httpx_aws_auth import AwsCredentials, AwsSigV4Auth
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ols import constants
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL_PREFIX = "anthropic."
OPENAI_MODEL_PREFIX = "openai."


@register_llm_provider_as(constants.PROVIDER_BEDROCK)
class Bedrock(LLMProvider):
    """AWS Bedrock provider using the Mantle gateway."""

    url: Optional[str] = None
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return default LLM params."""
        if self.provider_config.url is not None:
            self.url = str(self.provider_config.url).rstrip("/")
        self.credentials = self.provider_config.get_credentials()

        if not self.url:
            raise LLMConfigurationError(
                "url is required for Bedrock provider "
                "(e.g. https://bedrock-mantle.us-east-1.api.aws)"
            )

        if not self.credentials and not self._has_aws_credentials():
            raise LLMConfigurationError(
                "credentials are required for Bedrock provider — "
                "provide either a Bedrock API key (credentials_path file) "
                "or AWS IAM credentials (credentials_path directory with "
                "aws_access_key_id and aws_secret_access_key files)"
            )

        return {
            "api_key": self.credentials or "",
            "model": self.model,
            "temperature": 0.01,
            "max_tokens": 512,
        }

    def _has_aws_credentials(self) -> bool:
        """Check whether IAM credentials are available."""
        access_key, secret_key, _ = self.provider_config.get_aws_credentials()
        return access_key is not None and secret_key is not None

    def _region_from_url(self) -> str:
        """Extract AWS region from the Mantle gateway URL."""
        hostname = urlparse(self.url).hostname or ""
        parts = hostname.split(".")
        if len(parts) >= 3:
            return parts[1]
        raise LLMConfigurationError(
            f"cannot extract region from URL '{self.url}'; "
            "expected format: https://bedrock-mantle.<region>.api.aws"
        )

    def _build_boto3_session(self, region: str) -> boto3.Session:
        """Build a boto3 session from IAM credentials, with optional STS assume-role."""
        access_key, secret_key, role_arn = self.provider_config.get_aws_credentials()
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        if role_arn:
            sts = session.client("sts")
            assumed = sts.assume_role(
                RoleArn=role_arn, RoleSessionName="ols-bedrock"
            )["Credentials"]
            session = boto3.Session(
                aws_access_key_id=assumed["AccessKeyId"],
                aws_secret_access_key=assumed["SecretAccessKey"],
                aws_session_token=assumed["SessionToken"],
                region_name=region,
            )
        return session

    def _build_sigv4_auth(self, region: str) -> AwsSigV4Auth:
        """Build a SigV4 auth handler from IAM credentials."""
        session = self._build_boto3_session(region)
        frozen = session.get_credentials().get_frozen_credentials()
        creds = AwsCredentials(
            access_key=frozen.access_key,
            secret_key=frozen.secret_key,
            session_token=frozen.token,
        )
        return AwsSigV4Auth(credentials=creds, region=region, service="bedrock")

    def load(self) -> BaseChatModel:
        """Load LLM based on model prefix."""
        params = {**self.params}
        api_key = params.pop("api_key")
        model = params.pop("model")
        use_iam = self._has_aws_credentials()

        if model.startswith(ANTHROPIC_MODEL_PREFIX):
            params.pop("max_completion_tokens", None)
            region = self._region_from_url()
            region_prefix = region.split("-")[0]
            model_id = f"{region_prefix}.{model}"

            if use_iam:
                session = self._build_boto3_session(region)
                params["client"] = session.client("bedrock-runtime")
            else:
                params["bedrock_api_key"] = api_key

            params["model_id"] = model_id
            params["region_name"] = region
            return ChatBedrockConverse(**params)

        base_url = f"{self.url}/v1"
        use_responses_api = False

        if model.startswith(OPENAI_MODEL_PREFIX):
            base_url = f"{self.url}/openai/v1"
            use_responses_api = True

        max_tokens = params.pop("max_tokens", None)
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens

        params["model"] = model
        params["base_url"] = base_url
        params["use_responses_api"] = use_responses_api

        if use_iam:
            region = self._region_from_url()
            auth = self._build_sigv4_auth(region)
            params["openai_api_key"] = "unused"
            params["http_client"] = httpx.Client(auth=auth)
            params["http_async_client"] = httpx.AsyncClient(auth=auth)
        else:
            use_cert_store = self.provider_config.certificates_store is not None
            params["openai_api_key"] = api_key
            params["http_client"] = self._construct_httpx_client(use_cert_store, False)
            params["http_async_client"] = self._construct_httpx_client(
                use_cert_store, True
            )

        return ChatOpenAI(**params)
