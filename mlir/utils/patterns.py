import re


def process_single_pattern_block(block_text):
    """
    Processes a single log block, extracting relevant information and formatting it
    into a single, condensed, pipe-separated string.

    Args:
        block_text (str): A single block of log output (e.g., between two
        '===---' delimiters).

    Returns:
        str: The condensed log string if "pattern applied successfully" is found,
             otherwise an empty string.
    """

    # Initialize variables to store extracted data for this block
    pattern_name = ""
    processing_operation_type = ""
    # Using sets to store unique operation types
    inserted_operations = set()
    replaced_operations = set()
    modified_operations = set()
    erased_operations = set()
    pattern_applied_successfully_found = (
        False  # Flag to track if "pattern applied successfully" is found
    )

    # Split the block text into individual lines for processing
    lines = block_text.split("\n")

    for line in lines:
        # Regex to find the pattern name, e.g., "(anonymous namespace)::SimplifyAffineOp<...>"
        # It looks for lines starting with '* Pattern ' and captures everything until ' :'.
        # This regex also handles cases where there's no explicit name, capturing an empty string.
        pattern_match = re.search(r"^\s*\* Pattern (.*?) :", line)
        if pattern_match:
            inserted_operations = set()
            replaced_operations = set()
            modified_operations = set()
            erased_operations = set()
            pattern_name = pattern_match.group(1).strip()
            continue  # Move to the next line once found

        # Regex to find the main processing operation type, e.g., 'affine.apply'
        # It looks for lines starting with 'Processing operation : ' and captures the text
        # inside single quotes immediately after.
        processing_op_match = re.search(r"Processing operation : '(.*?)'", line)
        if processing_op_match:
            processing_operation_type = processing_op_match.group(1).strip()
            continue  # Move to the next line once found

        # Regex to find inserted, replaced, modified, or erased operations.
        # This matches lines starting with '** Insert', '** Replace', etc.,
        # and captures the action (Insert, Replace, Modified, Erase) and the
        # operation type (e.g., 'affine.apply', 'memref.load').
        op_detail_match = re.search(
            r"^\s*\*\* (Insert|Replace|Modified|Erase)\s*:\s*'(.*?)'\(", line
        )
        if op_detail_match:
            action = op_detail_match.group(1).strip().lower()
            op_type = op_detail_match.group(2).strip()

            # Add the extracted operation type to the corresponding set
            if action == "insert":
                inserted_operations.add(op_type)
            elif action == "replace":
                replaced_operations.add(op_type)
            elif action == "modified":
                modified_operations.add(op_type)
            elif action == "erase":
                erased_operations.add(op_type)
            continue  # Move to the next line once found

        # Check if the line contains the "pattern applied successfully" string
        if "pattern applied successfully" in line:
            pattern_applied_successfully_found = True

    # Only return the constructed string if "pattern applied successfully" was found
    if not pattern_applied_successfully_found:
      return ""

    # --- Construct the final output string for this block ---
    output_parts = []

    # Add the pattern name and processing operation type.
    output_parts.append(pattern_name)
    output_parts.append(processing_operation_type)

    # Helper function to format the sets of operations.
    def format_operations_set(ops_set):
        if ops_set:
            # Sort the operations for consistent output order
            return ", ".join(sorted(list(ops_set)))
        return ""

    # Format each category of operations and add to output_parts if not empty
    output_parts.append(format_operations_set(inserted_operations))
    output_parts.append(format_operations_set(replaced_operations))
    output_parts.append(format_operations_set(modified_operations))
    output_parts.append(format_operations_set(erased_operations))

    return " | ".join(output_parts)


def process_multiple_pattern_logs(full_log_text):
    """
    Splits a full log text into individual pattern blocks and processes each.
    Only includes output for blocks where "pattern applied successfully" is found,
    and ensures only unique lines are returned.

    Args:
        full_log_text (str): The complete multi-line log output string, potentially
                             containing multiple pattern application blocks.

    Returns:
        str: A multi-line string where each line represents a successfully
             processed and unique pattern block.
    """
    # Split the full log into individual blocks using the delimiter
    # re.split can produce empty strings at the start/end if delimiters are present there.
    raw_blocks = re.split(
        r"//===-------------------------------------------===//", full_log_text
    )

    output_lines = []
    seen_patterns = set()  # Set to store unique processed pattern lines

    for block in raw_blocks:
        block = block.strip()  # Clean up leading/trailing whitespace and empty lines
        if not block:  # Skip empty strings resulting from the split
            continue

        # Process each non-empty block
        condensed_line = process_single_pattern_block(block)

        # Only add to the final output if the block was successfully processed
        # and if this specific line hasn't been seen before.
        if condensed_line and condensed_line not in seen_patterns:
            output_lines.append(condensed_line)
            seen_patterns.add(condensed_line)

    output_lines.sort()
    return "\n".join(output_lines)


# --- Example Usage ---
if __name__ == "__main__":
    # Define a sample input with multiple blocks, including a repeated successful pattern
    sample_input_multi_block = """
//===-------------------------------------------===//
Processing operation : 'affine.apply'(0xSUCCESS1) {
  %3 = "affine.apply"(%arg0, %1) <{map = affine_map<(d0)[s0] -> (d0 - s0)>}> : (index, index) -> index

  * Pattern (anonymous namespace)::SimplifyAffineOp<mlir::affine.SuccessOp> : 'affine.apply -> ()' {
    ** Insert  : 'affine.insert.op1'(0x1111)
    ** Replace : 'affine.replace.op1'(0x2222)
  } -> success : pattern applied successfully
} -> success : at least one pattern matched
//===-------------------------------------------===//
Processing operation : 'memref.cast'(0xFAILED_BLOCK) {
  * Pattern (anonymous namespace)::FailedPattern : 'memref.cast -> ()' {
    ** Modified: 'memref.load'(0x3333)
  } -> fail : no pattern matched
} -> failure : all patterns failed to match
//===-------------------------------------------===//
Processing operation : 'affine.store'(0xSUCCESS2) {
  * Pattern (anonymous namespace)::AnotherSuccessfulPattern : 'affine.store -> ()' {
    ** Erase    : 'affine.erase.op2'(0x4444)
    ** Modified: 'memref.store'(0x5555)
  } -> success : pattern applied successfully
} -> success : at least one pattern matched
//===-------------------------------------------===//
Processing operation : 'test.simple'(0x55d876614200) {
  "test.simple"() : () -> ()

  * Pattern  : 'test.simple -> (test.success)' {
    ** Insert  : 'test.success'(0x55d876604410)
    ** Replace : 'test.simple'(0x55d876614200)
    ** Erase    : 'test.simple'(0x55d876614200)
  } -> success : pattern applied successfully
} -> success : at least one pattern matched
//===-------------------------------------------===//
Processing operation : 'affine.apply'(0xDUPLICATE_SUCCESS1) {
  %3 = "affine.apply"(%arg0, %1) <{map = affine_map<(d0)[s0] -> (d0 - s0)>}> : (index, index) -> index

  * Pattern (anonymous namespace)::SimplifyAffineOp<mlir::affine.SuccessOp> : 'affine.apply -> ()' {
    ** Insert  : 'affine.insert.op1'(0xAAAA)
    ** Replace : 'affine.replace.op1'(0xBBBB)
  } -> success : pattern applied successfully
} -> success : at least one pattern matched
//===-------------------------------------------===//
"""

    print("--- Original Multi-Block Input (with duplicate successful pattern) ---")
    print(sample_input_multi_block)

    converted_output = process_multiple_pattern_logs(sample_input_multi_block)
    print("\n--- Converted Output (Single line per unique successful pattern) ---")
    print(converted_output)
