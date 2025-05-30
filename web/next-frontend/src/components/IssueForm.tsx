"use client";

import { useState } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label"; // Assuming Label is available or will be added

interface IssueFormProps {
  onJobSubmit: (jobId: string) => void;
}

export default function IssueForm({ onJobSubmit }: IssueFormProps) {
  const [issueUrl, setIssueUrl] = useState<string>("");
  const [promptType, setPromptType] = useState<string>("explain");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError("");

    try {
      const response = await axios.post("http://localhost:8000/process_issue", {
        issue_url: issueUrl,
        prompt_type: promptType,
      });

      onJobSubmit(response.data.job_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to process issue");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 p-6 border rounded-lg shadow-lg bg-card text-card-foreground">
      <div className="space-y-2">
        <Label htmlFor="issue-url">GitHub Issue URL</Label>
        <Input
          id="issue-url"
          type="url"
          placeholder="https://github.com/owner/repo/issues/123"
          value={issueUrl}
          onChange={(e) => setIssueUrl(e.target.value)}
          required
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="prompt-type">Prompt Type</Label>
        <Select value={promptType} onValueChange={setPromptType}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select a prompt type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="explain">Explain Issue</SelectItem>
            <SelectItem value="fix">Suggest Fix</SelectItem>
            <SelectItem value="test">Generate Tests</SelectItem>
            <SelectItem value="summarize">Summarize</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Button type="submit" className="w-full" disabled={isSubmitting}>
        {isSubmitting ? "Processing..." : "Process Issue"}
      </Button>

      {error && <p className="text-destructive text-sm mt-2">{error}</p>}
    </form>
  );
}
