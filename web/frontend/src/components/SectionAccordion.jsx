import { useState } from 'react';

const SectionAccordion = ({ title, children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="accordion">
      <div 
        className="accordion-header" 
        onClick={() => setIsOpen(!isOpen)}
      >
        <h3>{title}</h3>
        <span className="accordion-icon">
          {isOpen ? '▼' : '►'}
        </span>
      </div>
      {isOpen && (
        <div className="accordion-content">
          {children}
        </div>
      )}
    </div>
  );
};

export default SectionAccordion;
