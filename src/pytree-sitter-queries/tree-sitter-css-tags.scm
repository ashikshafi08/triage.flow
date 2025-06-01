; CSS tags for code hierarchy parser

; CSS rule sets
(rule_set
  (selectors) @name.definition.rule_set) @definition.rule_set

; CSS at-rules (like @media, @keyframes)
(at_rule
  (at_keyword) @name.definition.at_rule
  (identifier)? @name.definition.at_rule) @definition.at_rule

; CSS keyframe blocks
(keyframe_block
  (percentage) @name.definition.keyframe_block) @definition.keyframe_block

; CSS declarations
(declaration
  (property_name) @name.definition.declaration) @definition.declaration
