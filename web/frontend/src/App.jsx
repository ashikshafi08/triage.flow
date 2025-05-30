import { useState, useEffect } from 'react';
import IssueForm from './components/IssueForm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
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

  return (
    <div className="app-container">
      <header>
        <h1>triage.flow</h1>
        <p>AI-powered GitHub Issue Analysis</p>
      </header>
      
      <main>
        <IssueForm onJobSubmit={handleNewJob} />
        
        {isLoading && <div className="loading">Processing issue... This may take 1-2 minutes</div>}
        
        {error && <div className="error">{error}</div>}
        
        {jobStatus?.status === 'completed' && (
          <div className="results">
            <h2>Analysis Results</h2>
            <div className="result-content">
              <SyntaxHighlighter 
                language="markdown" 
                style={atomDark}
                showLineNumbers
                wrapLongLines
              >
                {jobStatus.result}
              </SyntaxHighlighter>
            </div>
          </div>
        )}
        
        {jobStatus?.status === 'error' && (
          <div className="error">
            <h2>Processing Error</h2>
            <p>{jobStatus.error}</p>
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
