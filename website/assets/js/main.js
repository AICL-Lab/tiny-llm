// Tiny-LLM Site JavaScript

// Copy code buttons
document.addEventListener('DOMContentLoaded', function() {
  // Add copy buttons to code blocks
  document.querySelectorAll('pre.highlight').forEach(function(block) {
    const button = document.createElement('button');
    button.className = 'copy-code-button';
    button.textContent = 'Copy';
    button.setAttribute('aria-label', 'Copy code to clipboard');
    
    button.addEventListener('click', function() {
      const code = block.querySelector('code');
      if (code) {
        navigator.clipboard.writeText(code.textContent).then(function() {
          button.textContent = 'Copied!';
          setTimeout(function() {
            button.textContent = 'Copy';
          }, 2000);
        });
      }
    });
    
    block.style.position = 'relative';
    block.appendChild(button);
  });
});

// Copy button styles (injected via JS to avoid CSS file complexity)
const copyButtonStyles = document.createElement('style');
copyButtonStyles.textContent = `
  .copy-code-button {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary-color, #8b949e);
    background: var(--surface-secondary-color, #21262d);
    border: 1px solid var(--border-color, #30363d);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .copy-code-button:hover {
    background: var(--border-color, #30363d);
    color: var(--text-color, #f0f6fc);
  }
  
  pre.highlight:hover .copy-code-button {
    opacity: 1;
  }
`;
document.head.appendChild(copyButtonStyles);
