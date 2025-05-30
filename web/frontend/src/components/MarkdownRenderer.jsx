import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const MarkdownRenderer = ({ content }) => {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        children={content}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                children={String(children).trim()}
                style={atomDark}
                language={match[1]}
                PreTag="div"
                showLineNumbers
                {...props}
              />
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          h1: ({ node, ...props }) => <h1 className="heading-1" {...props} />,
          h2: ({ node, ...props }) => <h2 className="heading-2" {...props} />,
          h3: ({ node, ...props }) => <h3 className="heading-3" {...props} />,
          ul: ({ node, ...props }) => <ul className="list" {...props} />,
          ol: ({ node, ...props }) => <ol className="list" {...props} />,
          blockquote: ({ node, ...props }) => <blockquote className="blockquote" {...props} />,
          a: ({ node, ...props }) => <a className="link" target="_blank" rel="noopener noreferrer" {...props} />,
        }}
      />
    </div>
  );
};

export default MarkdownRenderer;
