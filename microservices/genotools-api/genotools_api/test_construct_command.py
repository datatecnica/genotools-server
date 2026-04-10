import sys
from unittest.mock import MagicMock

# Mock external packages so we can import utils without them installed
sys.modules["genotools"] = MagicMock()
sys.modules["genotools.utils"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.storage"] = MagicMock()

from genotools_api.models.models import GenoToolsParams
from genotools_api.utils.utils import construct_command


def test_hwe_included_when_set():
    params = GenoToolsParams(
        pfile="/tmp/test_geno",
        out="/tmp/test_out",
        hwe=1e-6,
    )
    cmd = construct_command(params)
    assert "--hwe 1e-06" in cmd


def test_hwe_excluded_when_none():
    params = GenoToolsParams(
        pfile="/tmp/test_geno",
        out="/tmp/test_out",
    )
    cmd = construct_command(params)
    assert "--hwe" not in cmd
