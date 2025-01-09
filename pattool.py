import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Optional, Type, Literal
from rag import vectorstore, retriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from openai import OpenAI
import os
import config

client = OpenAI()

class PatentSearchInput(BaseModel):
    lang: str = Field(
        description="language of query, default is English (en)", default="en"
    )
    query: str = Field(
        description="query string. For example, example 1: car and seat ; example 2: seat or wheel , example 3 (lang is cn): 汽车 and 座椅",
        default=None,
    )
    assignee_name: Optional[str] = Field(
        description="Assignee name. For example, example 1: Nintendo or Microsoft ; example 2: Philips",
        default=None,
    )


class PatentSearchTool(BaseTool):
    name: str = "patent_search_tool"
    description: str = (
        "Search for patents. Make sure the lang field and the language used for query are matched."
    )
    args_schema: Type[BaseModel] = PatentSearchInput
    return_direct: bool = True
    client_id: str = config.client_id
    client_secret: str = config.client_secret
    access_token: str = None
    token_url: str = "connect.zhihuiya.com/oauth/token"
    api_search_url: str = (
        "https://connect.zhihuiya.com/search/patent/query-search-patent"
    )
    api_file_url: str = "https://connect.zhihuiya.com//basic-patent-data/pdf-data"
    date_range_string: str = config.date_range_string

    def get_access_token(self):
        if self.access_token:
            return self.access_token
        request_url = f"https://{self.client_id}:{self.client_secret}@{self.token_url}"
        response = requests.request(
            "POST",
            request_url,
            data="grant_type=client_credentials",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        json_data = json.loads(response.text)
        token = json_data["data"]["token"]
        self.access_token = token
        return token

    def get_patent_list(
        self, query: str, lang: str = "en", assignee_name: str = None, limit: int = 5
    ):
        print("patent search: ", query, lang, assignee_name, limit)

        params = {"apikey": self.client_id}

        assignee_name_str = ""

        if assignee_name:
            assignee_name_str = f" and AN:({assignee_name})"

        payload = {
            "sort": [{"field": "PBDT_YEARMONTHDAY", "order": "DESC"}],
            "lang": lang,
            "limit": limit,
            "query_text": f"TA:({query}) {self.date_range_string} {assignee_name_str}",
            "offset": 0,
            "stemming": 0,
        }

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.get_access_token()}",
        }

        response = requests.request(
            "POST", self.api_search_url, params=params, json=payload, headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            print("data: ", data)
            if data.get("error_code", 0) != 0:
                return f"Error: {data}"
            return data.get("data", {})
        else:
            return f"Error: {response.status_code} - {response.text}"

    def get_patent_files(self, patent_ids: str):
        print("patent file: ", patent_ids)

        params = {
            "apikey": self.client_id,
            "patent_id": ",".join(patent_ids),
        }

        payload = None

        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.get_access_token()}",
        }

        response = requests.request(
            "GET", self.api_file_url, params=params, json=payload, headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("error_code", 0) != 0:
                return f"Error: {data}"
            return data.get("data", {})
        else:
            return f"Error: {response.status_code} - {response.text}"

    def download_patent_files(self, patents: list):
        print("download patent files: ", patents)

        import os

        if not os.path.exists("patent_files"):
            os.makedirs("patent_files")

        import requests

        for patent in patents:
            patent_id = patent.get("patent_id")
            if not patent_id:
                continue

            file = patent.get("file")
            if not file:
                continue

            pdf = file.get("pdf")
            if not pdf:
                continue

            path = pdf.get("path")

            if path:
                print("downloading: ", path)
                file_name = f"patent_files/{patent_id}.pdf"
                response = requests.get(path)
                if response.status_code == 200:
                    with open(file_name, "wb") as f:
                        f.write(response.content)

    def add_text_from_pdf(file_path):
        pdf_loader = UnstructuredPDFLoader(file_path)
        text = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(text)
        vectorstore.add_documents(documents=splits)

    def _run(
        self, query: str, lang: str = "en", assignee_name: str = None, limit: int = 10
    ):
        patent_list = self.get_patent_list(query, lang, assignee_name, limit)

        results = patent_list.get("results", [])

        patent_ids = [result.get("patent_id", "") for result in results]

        patent_files = self.get_patent_files(patent_ids)

        for result in results:
            for file in patent_files:
                if result.get("patent_id") == file.get("patent_id"):
                    result["file"] = file

        self.download_patent_files(results)

        file_paths = []
        for file_name in os.listdir("patent_files"):
            if file_name.endswith(".pdf"):
                file_paths.append(f"patent_files/{file_name}")
                self.add_text_from_pdf(f"patent_files/{file_name}")

        if config.sync_local_files_to_openai:
            if config.openai_vector_store_id is None:
                vector_store = client.beta.vector_stores.create(
                    name="design patents vector store"
                )
                config.openai_vector_store_id = vector_store.id

            file_streams = [open(path, "rb") for path in file_paths]
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=config.openai_vector_store_id, files=file_streams
            )

            print(file_batch.status)
            print(file_batch.file_counts)

        return results

    async def _arun(self, query: str):
        raise NotImplementedError("Async operation is not supported for this tool.")
