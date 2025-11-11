#!/usr/bin/env python3
import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify and print model information")
    embed_parser = subparsers.add_parser("embed_text", help="Create embedding of provided text")
    embed_parser.add_argument("text", help="text to embed")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings exist")
    embedquery_parser = subparsers.add_parser("embedquery", help="Embed provided query")
    embedquery_parser.add_argument("query", help="query to embed")
    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", help="query to search")
    search_parser.add_argument(
    "--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()