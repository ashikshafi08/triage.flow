"use client";

import { useState, useEffect } from "react";
import IssueForm from "@/components/IssueForm";
import MarkdownRenderer from "@/components/MarkdownRenderer"; // Will create this
import SectionAccordion from "@/components/SectionAccordion"; // Will create this
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area"; // Will create this
import { useToast } from "@/components/ui/use-toast"; // Will create this

interface ProgressLogEntry {
  timestamp: string;
  message: string;
}

export default function HomePage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const { toast } = useToast();

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8000/job_status/${jobId}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setJobStatus(data);

        if (data.status === "completed" || data.status === "error") {
          clearInterval(interval);
          setIsLoading(false);
          if (data.status === "completed") {
            toast({
              title: "Analysis Complete!",
              description: "Your GitHub issue has been processed.",
            });
          } else {
            toast({
              title: "Analysis Failed",
              description: data.error || "An unknown error occurred.",
              variant: "destructive",
            });
          }
        }
      } catch (err: any) {
        console.error("Failed to fetch job status:", err);
        toast({
          title: "Error",
          description: `Failed to fetch job status: ${err.message}`,
          variant: "destructive",
        });
        clearInterval(interval);
        setIsLoading(false);
      }
    }, 2000);

    setIsLoading(true);
    return () => clearInterval(interval);
  }, [jobId, toast]);

  const handleNewJob = (newJobId: string) => {
    setJobId(newJobId);
    setJobStatus(null);
    setIsLoading(true);
    toast({
      title: "Processing Started",
      description: "Your GitHub issue is being analyzed.",
    });
  };

  // Function to parse markdown into sections
  const parseMarkdownSections = (markdown: string) => {
    const sections: { title: string; content: string }[] = [];
    // Split by H2 or H3 headings
    const regex = /(#{2,3}\s.*)/g;
    const parts = markdown.split(regex);

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i].trim();
      if (part.startsWith("##") || part.startsWith("###")) {
        // This is a heading, the next part is its content
        sections.push({
          title: part.replace(/#{2,3}\s/, "").trim(),
          content: parts[i + 1] ? parts[i + 1].trim() : "",
        });
        i++; // Skip the content part as it's already processed
      } else if (i === 0 && part !== "") {
        // Handle content before the first heading as an introductory section
        sections.push({ title: "Introduction", content: part });
      }
    }
    return sections;
  };

  return (
    <div className="flex flex-col lg:flex-row gap-8">
      <div className="lg:w-2/3">
        <IssueForm onJobSubmit={handleNewJob} />

        {jobStatus?.status === "completed" && (
          <Card className="mt-8">
            <CardHeader>
              <CardTitle>Analysis Results</CardTitle>
            </CardHeader>
            <CardContent>
              {jobStatus.result && (
                <div className="space-y-4">
                  {parseMarkdownSections(jobStatus.result).map((section, index) => (
                    <SectionAccordion
                      key={index}
                      title={section.title}
                      defaultOpen={index === 0}
                    >
                      <MarkdownRenderer content={section.content} />
                    </SectionAccordion>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {jobStatus?.status === "error" && (
          <Card className="mt-8 border-destructive">
            <CardHeader>
              <CardTitle className="text-destructive">Processing Error</CardTitle>
            </CardHeader>
            <CardContent>
              <p>{jobStatus.error}</p>
            </CardContent>
          </Card>
        )}
      </div>

      {isLoading && (
        <div className="lg:w-1/3">
          <Card>
            <CardHeader>
              <CardTitle>Processing Status</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px] w-full rounded-md border p-4">
                <div className="space-y-2">
                  {jobStatus?.progress_log?.map((log: ProgressLogEntry, index: number) => (
                    <p key={index} className="text-sm text-muted-foreground">
                      <span className="font-semibold text-primary">
                        [{new Date(log.timestamp).toLocaleTimeString()}]
                      </span>{" "}
                      {log.message}
                    </p>
                  ))}
                </div>
              </ScrollArea>
              <p className="text-center text-sm text-muted-foreground mt-4">
                This may take 1-2 minutes...
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
