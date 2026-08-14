def custom_css(font_size="0.8rem"):
    css = [
        f"""
    /* Buttons */
    .bk-btn {{
        font-size: {font_size};
    }}

    /* Text and numeric input fields */
    input {{
        font-size: {font_size};
    }}

    /* Widget labels */
    .bk-input-group label {{
        font-size: {font_size};
    }}

    /* Slider labels and displayed values */
    .bk-slider-title,
    .bk-slider-value {{
        font-size: {font_size};
    }}

    /* General widget text */
    .bk-input-group {{
        font-size: {font_size};
    }}

    /* Panel Markdown */
    .bk-markdown,
    .markdown,
    .bk-clearfix {{
        font-size: {font_size};
    }}

    /* Markdown headings */
    .bk-markdown h1,
    .markdown h1 {{
        font-size: calc({font_size} * 2);
    }}

    .bk-markdown h2,
    .markdown h2 {{
        font-size: calc({font_size} * 1.5);
    }}

    .bk-markdown h3,
    .markdown h3 {{
        font-size: calc({font_size} * 1.25);
    }}

    /* Widget section headings */
    h4 {{
        font-size: calc({font_size} * 1.1);
    }}

    /* Checkbox / radio controls */
    .bk-checkbox,
    .bk-radio {{
        font-size: {font_size};
    }}

    /* Checkbox / radio text */
    input[type="checkbox"] + span,
    input[type="radio"] + span {{
        font-size: {font_size};
    }}

    /* Button group labels */
    .bk-btn-group label {{
        font-size: {font_size};
    }}
    """
    ]
    return css
