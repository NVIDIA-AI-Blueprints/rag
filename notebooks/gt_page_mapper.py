"""
Utility functions for document processing in RAG evaluation.

This module provides functionality to map ground truth contexts to their corresponding
document pages in a dataset. It processes various document formats (PDF, TXT) and
determines the most relevant pages using either explicit page numbers or text similarity.
"""

import os
import json
import argparse
import PyPDF2  # For PDF text extraction and page processing
from tqdm import tqdm  # Progress bar for long-running operations
from difflib import SequenceMatcher  # Text similarity calculation


# Configuration constants for dataset structure
# TODO: change based on your dataset location
DATASET_BASE_DIR = "../data"  # Base directory containing all evaluation datasets
EVAL_DATA = "data/financebench_open_source.jsonl"          # Standard filename for training/evaluation data
CORPUS_DIRECTORY = "pdfs"       # Directory containing document corpus files

def get_gt_file_pages_from_dataset(dataset_contexts, dataset_path):
    """
    Extract relevant document pages from a dataset based on context similarity.

    This function processes each context in the dataset to determine the most relevant
    document pages using three strategies:
    1. Use explicit ground truth page numbers if provided
    2. Find best matching page using text similarity for PDFs
    3. Return filename only if no page-level information is available

    Args:
        dataset_contexts (list): A list of context dictionaries from the dataset.
                                Each context should contain 'filename' and optionally
                                'gt_page' or 'text' fields.
        dataset_path (str): The path to the dataset directory containing document files.

    Returns:
        list: A list of relevant document identifiers with page numbers if applicable.
              Format: "filename.ext_pagenumber" or just "filename.ext"
    """
    relevant_docs = []
    try:
        # Determine file extension by examining first file in the corpus directory
        # This assumes all files in the corpus have the same extension
        files = os.listdir(dataset_path)
        if not files:
            return []  # Return empty list if no files found
        
        first_file = files[0]
        file_parts = first_file.rsplit(".", 1)  # Split on last dot to handle multiple dots
        file_ext = "." + file_parts[1] if len(file_parts) > 1 else ""  # Handle files without extension

        for context in dataset_contexts:
            # Strategy 1: Use explicit ground truth page numbers
            if "evidence_page_num" in context:
                # Ground truth page number is explicitly provided in the context
                if context["doc_name"].endswith((".pdf", ".txt")):
                    file_ext = ""  # Remove extension for files that already have it
                relevant_docs.append(context["doc_name"] + file_ext + "_" + str(context["evidence_page_num"]))
                
            # Strategy 2: Find best matching page using text similarity (PDF only)
            #elif "text" in context and context["text"] and file_ext == ".pdf":
            elif "evidence_text_full_page" in context and context["evidence_text_full_page"]:
                # For PDFs with ground truth text, find the most similar page
                if context["doc_name"].endswith((".pdf", ".txt")):
                    file_ext = ""  # Remove extension for files that already have it
                
                file_path = dataset_path + "/" + context["filename"] + file_ext
                try:
                    # Read PDF and compare each page with ground truth text
                    reader = PyPDF2.PdfReader(file_path)
                    best_match_page_number = 0  # Default to first page
                    max_similarity_ratio = 0
                    
                    for page_num, page in enumerate(reader.pages):
                        # Extract text from current page
                        page_text = page.extract_text()
                        
                        # Encode/decode to handle special characters consistently
                        page_text_normalized = page_text.encode('unicode_escape').decode()
                        
                        # Calculate similarity ratio between page text and ground truth
                        matcher = SequenceMatcher(None, page_text_normalized, context["evidence_text_full_page"])
                        similarity_ratio = matcher.ratio()
                        
                        # Keep track of the page with highest similarity
                        if similarity_ratio > max_similarity_ratio:
                            max_similarity_ratio = similarity_ratio
                            best_match_page_number = page_num
                            
                except Exception as e:
                    print(f"Error processing PDF {file_path}: {e}")
                    continue  # Skip this context if PDF processing fails
                
                relevant_docs.append(context["doc_name"] + file_ext + "_" + str(best_match_page_number))
                
            # Strategy 3: Use filename only (no page-level information available)
            else:
                # No ground truth page or text available, return filename only
                if context["doc_name"].endswith((".pdf", ".txt")):
                    file_ext = ""  # Remove extension for files that already have it
                relevant_docs.append(context["doc_name"] + file_ext)

        return relevant_docs
    
    except Exception as e:
        # Handle any unexpected errors during processing
        print(f"Failed getting the relevant docs and pages from the dataset, error: {e}")
        return None  # Return None to indicate processing failure


def main():
    """
    Main function to extract relevant document pages from a dataset and save them to a JSON file.

    This function orchestrates the complete workflow:
    1. Parse command-line arguments for dataset specification
    2. Load and validate dataset files
    3. Process each data point to extract relevant document pages
    4. Save results to an output JSON file with progress tracking

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract relevant documents with page numbers "
        "(if applicable) from dataset and save to a new JSON file")
    parser.add_argument(
        "--dataset", 
        required=True,
        help="Path to the dataset directory containing the corpus files and train.json"
    )
    
    args = parser.parse_args()

    # Construct paths to dataset files using standard directory structure
    dataset_file = os.path.join(DATASET_BASE_DIR, args.dataset, EVAL_DATA)  # Path to train.json
    dataset_path = os.path.join(DATASET_BASE_DIR, args.dataset, CORPUS_DIRECTORY)  # Path to corpus directory
    
    # Load and validate the dataset file
    try:
        with open(dataset_file, 'r') as f:
            dataset_data = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading dataset file {dataset_file}: {e}")
        return 1
    
    # Verify that the corpus directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist")
        return 1
    
    # Process each data point in the dataset with progress tracking
    relevant_docs = []
    with tqdm(total=len(dataset_data), desc="Getting relevant file pages", unit="datapoint") as pbar:
        for datapoint in dataset_data:
            contexts = []
            
            # Extract relevant file pages for the current data point
            files_pages = get_gt_file_pages_from_dataset(datapoint["evidence"], dataset_path)
            
            # Parse each file_page result to separate filename and page number
            for file_page in files_pages:
                # Check if the result includes a page number (contains underscore and doesn't end with .pdf/.txt)
                if not file_page.endswith((".pdf", ".txt")):
                    # Format: "filename.ext_pagenumber" - split to get filename and page
                    filename = file_page.rsplit("_", 1)[0]  # Everything before last underscore
                    page = file_page.rsplit("_", 1)[1]      # Page number after last underscore
                else:
                    # Format: "filename.ext" - no page number available
                    filename = file_page
                    page = ""  # Empty string indicates no specific page
                
                # Store the parsed filename and page information
                contexts.append({
                    "filename": filename,
                    "page": page,
                })
            
            # Create result entry for this data point
            relevant_docs.append({
                "original_id": datapoint["financebench_id"],  # Preserve original data point ID
                "contexts": contexts             # List of relevant file/page combinations
            })
            
            pbar.update(1)  # Update progress bar
    
    # Validate that processing was successful
    if not relevant_docs:
        print("Failed to process dataset")
        return 1
    
    # Save results to output file
    try:
        output_file = f"gt_file_pages-{args.dataset}.json"

        # Write the extracted relevant document pages to a JSON file with pretty formatting
        with open(output_file, 'w') as f:
            json.dump(relevant_docs, f, indent=2)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        return 1
    
    return 0  # Success exit code


if __name__ == "__main__":
    # Execute main function and exit with its return code
    exit(main())