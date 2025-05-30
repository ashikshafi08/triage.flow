import { useState } from 'react';
import axios from 'axios';

export default function IssueForm({ onJobSubmit }) {
  const [url, setUrl] = useState('');
  const [promptType, setPromptType] = useState('explain');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError('');
    
    try {
      const response = await axios.post('http://localhost:8000/process_issue', {
        issue_url: url,
        prompt_type: promptType
      });
      
      onJobSubmit(response.data.job_id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process issue');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="issue-form">
      <div className="form-group">
        <label htmlFor="issue-url">GitHub Issue URL</label>
        <input 
          type="url" 
          id="issue-url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://github.com/owner/repo/issues/123"
          required
          className="form-input"
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="prompt-type">Prompt Type</label>
        <select 
          id="prompt-type"
          value={promptType} 
          onChange={(e) => setPromptType(e.target.value)}
          className="form-select"
        >
          <option value="explain">Explain Issue</option>
          <option value="fix">Suggest Fix</option>
          <option value="test">Generate Tests</option>
          <option value="summarize">Summarize</option>
        </select>
      </div>
      
      <button 
        type="submit" 
        disabled={isSubmitting}
        className="submit-btn"
      >
        {isSubmitting ? 'Processing...' : 'Process Issue'}
      </button>
      
      {error && <p className="error-message">{error}</p>}
    </form>
  );
}
