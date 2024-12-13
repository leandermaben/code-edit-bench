prompt_to_apply_diff = {
    'user_prompt':"""
        Code changes will be provided in the form of a draft. You will need to apply the draft to the original code. 
        The original code will be enclosed within `<original_code>` tags.
        The draft will be enclosed within `<update_snippet>` tags.
        You need to output the update code within `<updated_code>` tags.

        Within the `<updated_code>` include only the final code after updation. Do not include any explanations or other content within these tags.
        
        <original_code>
        {original_code}
        </original_code>

        <update_snippet>
        {update_snippet}
        </update_snippet>
    """,
    'system_prompt':"""
        <updated_code>
        {updated_code}
        </updated_code>
    """
}

def get_messages(original_code, update_snippet, updated_code):
    return [
        {'role':'user', 'content':prompt_to_apply_diff['user_prompt'].format(original_code=original_code, update_snippet=update_snippet)},
        {'role':'assistant', 'content':prompt_to_apply_diff['system_prompt'].format(updated_code=updated_code)}
    ]

def get_messages_few_shot(original_code, update_snippet, updated_code):
    return [
        {'role':'user', 'content':prompt_to_apply_diff['user_prompt'].format(original_code=original_code, update_snippet=update_snippet)},
        {'role':'assistant', 'content':prompt_to_apply_diff['system_prompt'].format(updated_code=updated_code)}
    ]

# def get_messages(original_code, update_snippet, updated_code):
#     return [
#         {'role':'user', 'content':"You are a helpful assistant that applies code changes to the original code. You will be given the original code, the update snippet. You will need to apply the update snippet to the original code and return the updated code within `<updated_code>` tags. You DO NOT need to generate the original code or the update snippet."},
#         {'role':'user', 'content':prompt_to_apply_diff['user_prompt'].format(original_code=original_code, update_snippet=update_snippet)},
#         {'role':'assistant', 'content':prompt_to_apply_diff['system_prompt'].format(updated_code=updated_code)}
#     ]

def clean_code(code):
    return code.encode('utf-8',errors='ignore').decode('utf-8')

def extract_code_from_tags(text, tag="updated_code"):
    """
    Returns the original stripped text if the tag is not found.
    """
    # Find content between <updated_code> tags
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    
    if start_idx != -1 and end_idx != -1:
        # Add length of start tag to get content after it
        return text[start_idx + len(start_tag):end_idx].strip()
    return text.strip()  # Return original text if tags not found