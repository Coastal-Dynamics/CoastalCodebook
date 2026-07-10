def custom_css(font_size="16px"):
    css = [
        f"""
    .bk-btn {{
        font-size: {font_size};
    }}

    input {{
        font-size: {font_size};
    }}

    .bk-input-group label {{
        font-size: {font_size};
    }}

    .bk-markdown,
    .markdown,
    .bk-clearfix {{
        font-size: {font_size};
    }}

    /* Multiple choice / checkbox / radio labels */
    .bk-checkbox,
    .bk-radio {{
        font-size: {font_size};
    }}

    .bk-checkbox label,
    .bk-radio label,
    .bk-btn-group label {{
        font-size: {font_size};
    }}
    """
    ]

    return css
