from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.load.load import loads
from langchain_core.load.dump import dumps
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_community.embeddings import GPT4AllEmbeddings

import logging
logging.basicConfig()

from dotenv import load_dotenv
import os

load_dotenv()

groq_token = os.environ['HuggingFace_token']



# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))

# Multi Query Retriever Function for Query Translation.
def multi_query_retriever(llm, retriever, query):

    QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are a helpful assistant that generates multiple search queries based on a single input query. \n
                Generate multiple search queries related to: {question} \n
                Output (4 queries):""",
            )
    
    output_parser = LineListOutputParser()
    chain = QUERY_PROMPT | llm | output_parser

    multi_query = MultiQueryRetriever(
        retriever=retriever, llm_chain=chain, parser_key="lines"
    )

    #logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    return multi_query.invoke(query)

# RAG Fusion step to reranked a retrieved documents.
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results
    
def generation_step(llm,reranked_results,query):
    template = """Answer the following question based on this context:

                {context}

                Question: {question}
                """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = ( prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain.invoke({'context':reranked_results, "question": query})