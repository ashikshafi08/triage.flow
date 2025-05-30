import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "triage.flow",
  description: "AI-powered GitHub Issue Analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">
        <div className="flex flex-col min-h-screen">
          <header className="py-4 border-b border-border">
            <div className="container mx-auto px-4 text-center">
              <h1 className="text-3xl font-bold text-primary">triage.flow</h1>
              <p className="text-muted-foreground">AI-powered GitHub Issue Analysis</p>
            </div>
          </header>
          <main className="flex-grow container mx-auto px-4 py-8">
            {children}
          </main>
          <footer className="py-4 border-t border-border text-center text-muted-foreground text-sm">
            <div className="container mx-auto px-4">
              <p>Powered by triage.flow | MIT License</p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
