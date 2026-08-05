"""可选依赖的错误边界。"""

from unittest.mock import patch

import pytest

from ops.pipeline_impl import _load_point_cloud_utils


def test_missing_point_cloud_utils_has_install_hint():
    missing = ModuleNotFoundError(
        "No module named 'point_cloud_utils'",
        name="point_cloud_utils",
    )

    with patch("builtins.__import__", side_effect=missing):
        with pytest.raises(RuntimeError, match="需要安装 point-cloud-utils"):
            _load_point_cloud_utils()


def test_broken_point_cloud_utils_keeps_load_failure():
    broken = ImportError("DLL load failed")

    with patch("builtins.__import__", side_effect=broken):
        with pytest.raises(RuntimeError, match="已安装.*二进制依赖无法加载"):
            _load_point_cloud_utils()
