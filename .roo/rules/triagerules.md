---
description: 
globs: 
alwaysApply: false
---
# Roo Code Rules for Triage.Flow Project

## Project Overview
This project consists of two main components:
- **Backend (src/)**: FastAPI-based Python application with AI/LLM integration, GitHub API client, RAG systems, and agentic tools
- **Frontend (issue-flow-ai-prompt/)**: React + TypeScript + Vite application using shadcn/ui components

## General Guidelines
- Follow existing code patterns and conventions within each component
- Prioritize readability and maintainability
- Use meaningful variable and function names
- Add appropriate type hints and documentation
- Handle errors gracefully with proper logging

## Python Backend (src/) Rules

### Code Style & Structure
- Use Python 3.8+ features
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Prefer dataclasses or Pydantic models for data structures
- Use relative imports (e.g., `from .config import settings`)
- Place imports in this order: standard library, third-party, local imports

### FastAPI Patterns
- Use dependency injection for shared resources (session_manager, etc.)
- Group related endpoints in router modules under `api/routers/`
- Use Pydantic models for request/response validation
- Include proper HTTP status codes and error responses
- Add descriptive docstrings for API endpoints

### Configuration & Settings
- Use the existing `Settings` class in `config.py` for all configuration
- Access settings via `from .config import settings`
- Support environment variables with sensible defaults
- Use feature flags for experimental functionality

### LLM & AI Integration
- Use the existing `LLMClient` for all AI model interactions
- Support multiple providers (OpenAI, OpenRouter) via configuration
- Implement proper token counting and context window management
- Use caching for expensive LLM operations when appropriate
- Handle rate limiting and API errors gracefully

### Data Models
- Define all data structures in `models.py` using Pydantic
- Use proper type annotations and validation
- Include example values in docstrings when helpful
- Maintain backward compatibility when modifying existing models

### Caching & Performance
- Use the existing cache managers (`rag_cache`, `response_cache`, `folder_cache`)
- Implement cache keys that are deterministic and collision-free
- Set appropriate TTL values based on data volatility
- Monitor cache hit rates and memory usage

### Agentic Tools & RAG
- Follow the existing tool pattern in `agent_tools/`
- Use the `FunctionTool` base class for new tools
- Implement proper error handling in tool functions
- Maintain tool registry for discoverability
- Use chunking strategies for large content processing

### Error Handling & Logging
- Use Python's logging module with appropriate levels
- Include context in error messages (file paths, user inputs, etc.)
- Catch specific exceptions rather than broad Exception catches
- Return structured error responses with helpful messages

### Testing & Quality
- Write unit tests for new functionality
- Mock external dependencies (GitHub API, LLM providers)
- Use pytest fixtures for common test setups
- Test both success and error cases

## React Frontend (issue-flow-ai-prompt/) Rules

### Code Style & Structure
- Use TypeScript with strict mode enabled
- Prefer functional components with hooks
- Use the `@/` alias for imports (configured in tsconfig)
- Group imports: React, third-party, local components, types/interfaces

### Component Patterns
- Use shadcn/ui components as building blocks
- Create reusable components in `components/` directory
- Use proper TypeScript interfaces for props
- Implement proper error boundaries for robust UIs

### State Management
- Use React Query (@tanstack/react-query) for server state
- Use React hooks (useState, useEffect) for local state
- Implement proper loading and error states
- Cache API responses appropriately

### Styling & UI
- Use Tailwind CSS for styling with existing utility classes
- Follow the existing design system from shadcn/ui
- Use consistent spacing, colors, and typography
- Implement responsive design patterns
- Use Framer Motion for animations when appropriate

### Routing & Navigation
- Use React Router for client-side routing
- Follow RESTful URL patterns where possible
- Implement proper 404 handling
- Use React Router's type-safe navigation

### API Integration
- Create custom hooks for API calls using React Query
- Handle loading, error, and success states consistently
- Implement proper error messages and user feedback
- Use TypeScript interfaces matching backend models

### Forms & Validation
- Use React Hook Form with Zod validation
- Implement proper form error handling and display
- Use controlled components for form inputs
- Provide clear validation feedback to users

### Performance
- Use React.memo for expensive components
- Implement proper loading states to prevent layout shift
- Optimize bundle size with dynamic imports where appropriate
- Use proper keys for list rendering

### Code Organization
- Place page components in `pages/` directory
- Group related components in feature-specific folders
- Use barrel exports (index.ts) for cleaner imports
- Keep components small and focused on single responsibilities

## Cross-Cutting Concerns

### API Communication
- Maintain consistency between frontend TypeScript types and backend Pydantic models
- Use proper HTTP methods (GET, POST, PUT, DELETE)
- Implement proper request/response logging
- Handle authentication and authorization consistently

### Error Handling
- Provide user-friendly error messages
- Log detailed error information for debugging
- Implement graceful degradation for non-critical features
- Use proper HTTP status codes

### Security
- Validate all inputs on both frontend and backend
- Use environment variables for sensitive configuration
- Implement proper CORS policies
- Sanitize user-generated content

### Documentation
- Add JSDoc comments for complex functions
- Maintain up-to-date README files
- Document API endpoints with proper examples
- Include type information in all interfaces

## File Naming Conventions
- Python: snake_case for files and variables, PascalCase for classes
- TypeScript: PascalCase for components, camelCase for functions/variables
- Use descriptive names that indicate the file's purpose
- Group related files in appropriate directories

## Dependencies
- Keep dependencies up to date but test thoroughly before upgrading
- Prefer well-maintained libraries with good TypeScript support
- Document any custom or complex dependencies
- Use exact versions for critical dependencies

## Environment & Configuration
- Use `.env` files for local development
- Never commit sensitive information to version control
- Provide example environment files (`.env.example`)
- Use different configurations for development/production

Remember: When in doubt, follow the existing patterns in the codebase and prioritize consistency over personal preferences. 