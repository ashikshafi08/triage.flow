; Markdown tags for code hierarchy parser

; Markdown headings
(atx_heading
  (atx_heading_marker)
  (heading_content) @name.definition.heading) @definition.heading

; Markdown code blocks
(fenced_code_block
  (info_string)? @name.definition.code_block) @definition.code_block

; Markdown lists
(list
  (list_item) @name.definition.list_item) @definition.list

; Markdown links
(link
  (link_text) @name.definition.link) @definition.link

; Markdown blockquotes
(block_quote) @definition.block_quote
