#!/usr/bin/env python3
"""
Test script for Precision Medicine Carriers Pipeline API.

Tests API endpoints with sample requests.
"""

import requests
import time
import json
from typing import Dict, Any


API_BASE_URL = "http://localhost:8000/api/v1/carriers"


def test_health_check():
    """Test health check endpoint."""
    print("🔍 Testing health check endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")

    print(f"   Status code: {response.status_code}")
    print(f"   Response: {response.json()}")

    assert response.status_code == 200, "Health check failed"
    print("   ✅ Health check passed\n")


def test_pipeline_submission():
    """Test pipeline submission endpoint."""
    print("🔍 Testing pipeline submission...")

    # Create a test request (small subset for testing)
    request_data = {
        "job_name": "api_test_job",
        "ancestries": ["AAC"],  # Just one ancestry for quick test
        "data_types": ["NBA"],  # Just NBA for quick test
        "parallel": True,
        "optimize": True,
        "skip_extraction": False,
        "skip_probe_selection": True,  # Skip to reduce test time
        "skip_locus_reports": True  # Skip to reduce test time
    }

    print(f"   Request: {json.dumps(request_data, indent=2)}")

    response = requests.post(
        f"{API_BASE_URL}/pipeline",
        json=request_data
    )

    print(f"   Status code: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Pipeline submission failed"

    result = response.json()
    assert result["success"], "Pipeline submission reported failure"
    assert "job_id" in result, "No job_id in response"

    job_id = result["job_id"]
    print(f"   ✅ Pipeline submitted with job_id: {job_id}\n")

    return job_id


def test_job_status(job_id: str, poll_interval: int = 5, max_polls: int = 240):
    """
    Test job status endpoint and poll until completion.

    Args:
        job_id: Job identifier
        poll_interval: Seconds between status checks
        max_polls: Maximum number of polls (240 * 5s = 20 minutes)
    """
    print(f"🔍 Testing job status endpoint for job {job_id}...")

    for i in range(max_polls):
        response = requests.get(f"{API_BASE_URL}/pipeline/{job_id}")

        assert response.status_code == 200, f"Job status check failed: {response.status_code}"

        status = response.json()
        print(f"   [{i+1}/{max_polls}] Status: {status['status']} - {status.get('progress', 'N/A')}")

        if status["status"] in ["completed", "failed"]:
            print(f"   ✅ Job finished with status: {status['status']}\n")
            return status

        time.sleep(poll_interval)

    print(f"   ⚠️  Job did not complete within {max_polls * poll_interval}s\n")
    return None


def test_job_results(job_id: str):
    """Test job results endpoint."""
    print(f"🔍 Testing job results endpoint for job {job_id}...")

    response = requests.get(f"{API_BASE_URL}/pipeline/{job_id}/results")

    if response.status_code == 200:
        result = response.json()
        print(f"   Status code: {response.status_code}")
        print(f"   Success: {result.get('success')}")
        print(f"   Execution time: {result.get('execution_time_seconds')}s")
        print(f"   Output files: {len(result.get('output_files', {}))} files")
        print(f"   Summary: {result.get('summary', {})}")
        print("   ✅ Results retrieved successfully\n")
        return result
    elif response.status_code == 400:
        print(f"   ⚠️  Job not completed yet (status code 400)")
        print(f"   Response: {response.json()}\n")
        return None
    else:
        print(f"   ❌ Failed with status code: {response.status_code}")
        print(f"   Response: {response.json()}\n")
        return None


def main():
    """Run all API tests."""
    print("=" * 70)
    print("Precision Medicine Carriers Pipeline API Tests")
    print("=" * 70)
    print()

    try:
        # Test 1: Health check
        test_health_check()

        # Test 2: Pipeline submission
        job_id = test_pipeline_submission()

        # Test 3: Job status polling
        final_status = test_job_status(job_id, poll_interval=10, max_polls=120)

        # Test 4: Get results
        if final_status and final_status["status"] == "completed":
            test_job_results(job_id)

        print("=" * 70)
        print("✅ All API tests completed!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Is the server running?")
        print("   Start the API with: ./run_api.sh")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
