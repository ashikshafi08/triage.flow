import { useState, useEffect } from 'react';
import IssueForm from './components/IssueForm';
import MarkdownRenderer from './components/MarkdownRenderer';
import SectionAccordion from './components/SectionAccordion';
import axios from 'axios';
import './App.css';

function App() {
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Poll job status when jobId changes
  useEffect(() => {
    if (!jobId) return;
    
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:8000/job_status/${jobId}`);
        setJobStatus(response.data);
        
        // Stop polling if job is completed or failed
        if (response.data.status === 'completed' || response.data.status === 'error') {
          clearInterval(interval);
          setIsLoading(false);
        }
      } catch (err) {
        setError('Failed to fetch job status');
        clearInterval(interval);
        setIsLoading(false);
      }
    }, 2000);
    
    setIsLoading(true);
    return () => clearInterval(interval);
  }, [jobId]);

  const handleNewJob = (newJobId) => {
    setJobId(newJobId);
    setJobStatus(null);
    setError('');
  };

  // Function to parse markdown into sections
  const parseMarkdownSections = (markdown) => {
    const sections = [];
    // Split by H2 or H3 headings
    const regex = /(#{2,3}\s.*)/g;
    const parts = markdown.split(regex);

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i].trim();
      if (part.startsWith('##') || part.startsWith('###')) {
        // This is a heading, the next part is its content
        sections.push({
          title: part.replace(/#{2,3}\s/, '').trim(),
          content: parts[i + 1] ? parts[i + 1].trim() : ''
        });
        i++; // Skip the content part as it's already processed
      } else if (i === 0 && part !== '') {
        // Handle content before the first heading as an introductory section
        sections.push({ title: 'Introduction', content: part });
      }
    }
    return sections;
  };

  return (
    <div className="app-container">
      <header>
        <h1>triage.flow</h1>
        <p>AI-powered GitHub Issue Analysis</p>
      </header>
      
      <main className={`main-content ${isLoading ? 'two-panel' : 'one-panel'}`}>
        <div className="left-panel">
          <IssueForm onJobSubmit={handleNewJob} />
          
          {error && <div className="error">{error}</div>}
          
          {jobStatus?.status === 'completed' && (
            <div className="results">
              <h2>Analysis Results</h2>
              {/* Render structured results */}
              {jobStatus.result && (
                <div className="structured-results">
                  {parseMarkdownSections(jobStatus.result).map((section, index) => (
                    <SectionAccordion 
                      key={index} 
                      title={section.title} 
                      defaultOpen={index === 0} // Open the first section by default
                    >
                      <MarkdownRenderer content={section.content} />
                    </SectionAccordion>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {jobStatus?.status === 'error' && (
            <div className="error">
              <h2>Processing Error</h2>
              <p>{jobStatus.error}</p>
            </div>
          )}
        </div>

        {isLoading && (
          <div className="right-panel">
            <div className="loading-panel">
              <h3>Processing Status</h3>
              <div className="progress-log">
                {jobStatus?.progress_log?.map((log, index) => (
                  <p key={index} className="log-entry">
                    <span className="timestamp">[{new Date(log.timestamp).toLocaleTimeString()}]</span> {log.message}
                  </p>
                ))}
              </div>
              <p className="loading-message">This may take 1-2 minutes...</p>
            </div>
          </div>
        )}
      </main>
      
      <footer>
        <p>Powered by triage.flow | MIT License</p>
      </footer>
    </div>
  );
}

export default App;
