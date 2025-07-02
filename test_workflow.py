# test_workflow.py
import logging
import sys
from src.agent_tools.llamaindex_comprehensive_workflow import ComprehensiveAnalysisWorkflow
from src.agent_tools.context_manager import ContextManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Test the workflow directly
try:
    print('Testing comprehensive workflow...')
    import asyncio
    from pathlib import Path

    async def test_workflow():
        repo_path = '/Users/ash/Documents/ash_projects/triage.flow'
        context_manager = ContextManager(None, None)

        workflow = ComprehensiveAnalysisWorkflow(
            session_id='test',
            repo_path=repo_path,
            context_manager=context_manager
        )

        from src.agent_tools.llamaindex_comprehensive_workflow import AnalysisRequest
        request = AnalysisRequest(
            query='Analyze security vulnerabilities',
            focus_areas=['security', 'dependencies'],
            session_id='test',
            repo_path=repo_path
        )

        result = await workflow.run(request=request)
        print('Workflow completed successfully!')
        print('Result type:', type(result))
        print('Result keys:', list(result.keys()) if isinstance(result, dict) else 'Not a dict')
        
        # Print some key sections to verify the structure
        if isinstance(result, dict):
            if 'analysis_metadata' in result:
                print('Analysis metadata keys:', list(result['analysis_metadata'].keys()))
            if 'security_analysis' in result:
                print('Security findings count:', result['security_analysis'].get('findings_count', 0))
            if 'summary' in result:
                print('Summary preview:', result['summary'][:200] + '...' if len(result['summary']) > 200 else result['summary'])
        
        return result

    result = asyncio.run(test_workflow())
    print('Test completed')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
