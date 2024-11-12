def format_paragraph(res):
    if 'paragraph_text' in res:
        rtext = res['paragraph_text']
    elif 'text' in res:
        if 'title' in res:
            rtext = f"The title is {res['title']} and the paragraph is {res['text']}"
        else:
            rtext = f"The paragraph is {res['text']}"
    else:
        print(f"Warning: retrieve result does not contain paragraph_text or text: {res}")
        rtext = ''
    return rtext
