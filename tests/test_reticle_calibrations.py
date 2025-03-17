# -*- coding: utf-8 -*-
"""
Test suite for reticle_calibrations.py
"""

import logging
import unittest
from pathlib import Path

import numpy as np

from aind_mri_utils.reticle_calibrations import (
    combine_parallax_and_manual_calibrations,
    debug_manual_calibration,
    debug_parallax_and_manual_calibrations,
    debug_parallax_calibration,
    find_probe_angle,
    find_probe_insertion_vector,
    fit_rotation_params,
    fit_rotation_params_from_manual_calibration,
    fit_rotation_params_from_parallax,
    read_manual_reticle_calibration,
    read_parallax_calibration_dir,
    read_parallax_calibration_dir_and_correct,
    read_parallax_calibration_file,
    transform_bregma_to_probe,
    transform_bregma_to_reticle,
    transform_probe_to_bregma,
    transform_reticle_to_bregma,
)

logger = logging.getLogger("aind_mri_utils.reticle_calibrations")
logger.addHandler(logging.NullHandler())


class CalibrationTest(unittest.TestCase):
    test_data_dir = Path(__file__).parent / "../test-data/"
    reticle_data_path = test_data_dir / "reticle"
    manual_calibration_file = (
        reticle_data_path / "calibration_info_np2_2025_03_11T13_42_00.xlsx"
    )
    parallax_calibration_path = reticle_data_path / "log_20250311_110408"
    parallax_example_file = (
        parallax_calibration_path / "points_SN46105_20250311_111453.csv"
    )
    man_calibration_pts = {
        46116: (
            np.array(
                [
                    [0.076, 0.062, 0.311],
                    [-5.924, 0.062, 0.311],
                    [2.076, 0.062, 0.311],
                    [0.076, 2.062, 0.311],
                    [0.076, -3.938, 0.311],
                ]
            ),
            np.array(
                [
                    [5.554, 8.504, 6.509],
                    [11.025, 8.501, 4.016],
                    [3.729, 8.499, 7.326],
                    [5.784, 10.455, 7.029],
                    [5.078, 4.603, 5.467],
                ]
            ),
        ),
        46100: (
            np.array(
                [
                    [0.076, 0.062, 0.311],
                    [-5.924, 0.062, 0.311],
                    [2.076, 0.062, 0.311],
                    [0.076, 2.062, 0.311],
                    [0.076, -3.938, 0.311],
                ]
            ),
            np.array(
                [
                    [7.035, 10.515, 5.478],
                    [12.648, 10.423, 3.189],
                    [5.166, 10.538, 6.226],
                    [7.087, 12.536, 5.494],
                    [6.994, 6.52, 5.421],
                ]
            ),
        ),
    }
    parallax_calibration_pts = {
        46105: (
            np.array(
                [
                    [-1122.0, -1792.0, 720.0],
                    [-1978.0, -1602.0, 580.0],
                    [-2322.0, -1525.0, 549.0],
                    [-2575.0, -1421.0, 508.0],
                    [-2808.0, -1398.0, 452.0],
                    [-2805.0, -1302.0, 495.0],
                    [-3168.0, -1294.0, 414.0],
                    [-3372.0, -1388.0, 337.0],
                    [-2831.0, -1256.0, 580.0],
                    [-2920.0, 183.0, 996.0],
                    [-2914.0, 271.0, 1027.0],
                    [-3186.0, 289.0, 967.0],
                    [-3637.0, 393.0, 386.0],
                    [-3083.0, 228.0, 1033.0],
                    [-2098.0, 607.0, 525.0],
                    [-2096.0, 755.0, 573.0],
                    [-1732.0, 751.0, 644.0],
                    [-1177.0, 851.0, 298.0],
                    [-905.0, 635.0, 536.0],
                    [-602.0, 771.0, 88.0],
                ]
            ),
            np.array(
                [
                    [9761.5, 5104.0, 5955.5],
                    [8889.5, 4947.0, 5955.0],
                    [8520.5, 4854.5, 5955.5],
                    [8265.0, 4765.0, 5955.5],
                    [8017.5, 4764.5, 5955.5],
                    [8017.5, 4653.5, 5955.5],
                    [7656.0, 4653.0, 5955.5],
                    [7445.0, 4764.5, 5955.0],
                    [8020.0, 4590.0, 5878.5],
                    [7955.0, 3069.0, 5878.0],
                    [7955.0, 2970.0, 5878.5],
                    [7688.0, 2970.0, 5878.5],
                    [7113.0, 3011.5, 6364.5],
                    [7810.0, 3011.5, 5826.0],
                    [8652.5, 2800.0, 6601.0],
                    [8652.5, 2643.0, 6600.0],
                    [9030.0, 2643.0, 6600.5],
                    [9501.0, 2643.0, 7071.0],
                    [9827.5, 2785.5, 6859.5],
                    [10025.5, 2801.5, 7373.5],
                ]
            ),
        ),
    }
    parallax_corrected_calibration_pts = {
        46105: (
            np.array(
                [
                    [-1.046, -1.73, 1.031],
                    [-1.902, -1.54, 0.891],
                    [-2.246, -1.463, 0.86],
                    [-2.499, -1.359, 0.819],
                    [-2.732, -1.336, 0.763],
                    [-2.729, -1.24, 0.806],
                    [-3.092, -1.232, 0.725],
                    [-3.296, -1.326, 0.648],
                    [-2.755, -1.194, 0.891],
                    [-2.844, 0.245, 1.307],
                    [-2.838, 0.333, 1.338],
                    [-3.11, 0.351, 1.278],
                    [-3.561, 0.455, 0.697],
                    [-3.007, 0.29, 1.344],
                    [-2.022, 0.669, 0.836],
                    [-2.02, 0.817, 0.884],
                    [-1.656, 0.813, 0.955],
                    [-1.101, 0.913, 0.609],
                    [-0.829, 0.697, 0.847],
                    [-0.526, 0.833, 0.399],
                ]
            ),
            np.array(
                [
                    [9.7615, 5.104, 5.9555],
                    [8.8895, 4.947, 5.955],
                    [8.5205, 4.8545, 5.9555],
                    [8.265, 4.765, 5.9555],
                    [8.0175, 4.7645, 5.9555],
                    [8.0175, 4.6535, 5.9555],
                    [7.656, 4.653, 5.9555],
                    [7.445, 4.7645, 5.955],
                    [8.02, 4.59, 5.8785],
                    [7.955, 3.069, 5.878],
                    [7.955, 2.97, 5.8785],
                    [7.688, 2.97, 5.8785],
                    [7.113, 3.0115, 6.3645],
                    [7.81, 3.0115, 5.826],
                    [8.6525, 2.8, 6.601],
                    [8.6525, 2.643, 6.6],
                    [9.03, 2.643, 6.6005],
                    [9.501, 2.643, 7.071],
                    [9.8275, 2.7855, 6.8595],
                    [10.0255, 2.8015, 7.3735],
                ]
            ),
        )
    }
    man_cal_errs = {
        46116: np.array(
            [0.00529405, 0.00310946, 0.0082168, 0.00732138, 0.00295256]
        ),
    }
    par_cal_errs = {
        46105: np.array(
            [
                0.00922202,
                0.01959963,
                0.01796559,
                0.00501619,
                0.01202905,
                0.01168256,
                0.00956649,
                0.00921835,
                0.00682431,
                0.0053811,
                0.01085676,
                0.0072889,
                0.01432768,
                0.01074016,
                0.00512335,
                0.0052497,
                0.01315437,
                0.007188,
                0.01215161,
                0.00730074,
            ]
        ),
    }
    global_offset = np.array([0.076, 0.062, 0.311])
    global_rotation_degrees = 0
    reticle_name = "H"

    manual_test_pairs = {
        46116: (
            np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
            np.array([[4.55, 9.62, 6.53], [6.93, 7.09, 6.93]]),
        ),
    }
    parallax_test_pairs = {
        46105: (
            np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
            np.array([[11.65, 2.49, 7.17], [9.35, 4.97, 8.06]]),
        ),
    }

    def setUp(self):
        # Store original logging configuration to restore later
        self.root_logger = logging.getLogger()
        self.old_level = self.root_logger.level
        self.old_handlers = self.root_logger.handlers.copy()

        # Configure logging to capture but not display output
        # 1. Set level to DEBUG so statements execute
        # 2. Remove any existing handlers
        # 3. Add a NullHandler to prevent output
        self.root_logger.setLevel(logging.DEBUG)
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        self.root_logger.addHandler(logging.NullHandler())

    def tearDown(self):
        # Restore original logging configuration
        self.root_logger.setLevel(self.old_level)
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        for handler in self.old_handlers:
            self.root_logger.addHandler(handler)

    def helper_test_transforms(
        self, R, t, bregma_pt, probe_pt, atol=1e-05, rtol=1e-08
    ):
        received_probe_pt = transform_bregma_to_probe(bregma_pt, R, t)
        self.assertTrue(
            np.allclose(received_probe_pt, probe_pt, atol=atol, rtol=rtol)
        )
        received_bregma_pt = transform_probe_to_bregma(probe_pt, R, t)
        self.assertTrue(
            np.allclose(received_bregma_pt, bregma_pt, atol=atol, rtol=rtol)
        )

    def helper_test_calibration(
        self, cal_by_probe, test_pair_by_probe, *args, **kwargs
    ):
        for probe, (bregma_pt, probe_pt) in test_pair_by_probe.items():
            R, t, _ = cal_by_probe[probe]
            self.helper_test_transforms(
                R, t, bregma_pt, probe_pt, *args, **kwargs
            )

    def test_fit_rotation_params(self) -> None:
        """Tests for fit_rotation_params"""
        reticle_pts, probe_pts = self.parallax_calibration_pts[46105]
        reticle_pts_scaled = reticle_pts / 1000
        probe_pts_scaled = probe_pts / 1000
        reticle_pts_scaled = reticle_pts_scaled + self.global_offset
        for find_scaling in [True, False]:
            if find_scaling:
                atol = 1e-2
                rtol = 1e-2
            else:  # no scaling
                atol = 1e-1
                rtol = 0
            R, t, s = fit_rotation_params(
                reticle_pts_scaled,
                probe_pts_scaled,
                find_scaling=find_scaling,
            )
            self.helper_test_transforms(
                R, t, *self.parallax_test_pairs[46105], atol=atol, rtol=rtol
            )

    def test_read_manual_reticle_calibration(self) -> None:
        """Tests for read_manual_reticle_calibration"""
        (
            adjusted_pairs_by_probe,
            global_offset,
            global_rotation_degrees,
            reticle_name,
        ) = read_manual_reticle_calibration(self.manual_calibration_file)
        for k, v in self.man_calibration_pts.items():
            self.assertTrue(
                np.allclose(v[0], adjusted_pairs_by_probe[k][0])
                and np.allclose(v[1], adjusted_pairs_by_probe[k][1])
            )
        self.assertTrue(
            np.allclose(global_offset, self.global_offset)
            and global_rotation_degrees == self.global_rotation_degrees
            and reticle_name == self.reticle_name
        )

    def test_fit_rotation_params_from_manual_calibration(self) -> None:
        """Tests for fit_rotation_params_from_manual_calibration"""
        cal_by_probe, R_reticle_to_bregma, global_offset = (
            fit_rotation_params_from_manual_calibration(
                self.manual_calibration_file
            )
        )
        self.helper_test_calibration(
            cal_by_probe, self.manual_test_pairs, atol=1e-2
        )
        self.assertTrue(np.array_equal(R_reticle_to_bregma, np.eye(3)))
        self.assertTrue(np.array_equal(global_offset, self.global_offset))

    def test_debug_manual_calibration(self) -> None:
        """Tests for fit_rotation_params_from_manual_calibration"""
        (
            cal_by_probe,
            R_reticle_to_bregma,
            t_reticle_to_bregma,
            adjusted_pairs_by_probe,
            errs_by_probe,
        ) = debug_manual_calibration(self.manual_calibration_file)

        self.helper_test_calibration(
            cal_by_probe, self.manual_test_pairs, atol=1e-2
        )
        self.assertTrue(np.array_equal(R_reticle_to_bregma, np.eye(3)))
        self.assertTrue(
            np.array_equal(t_reticle_to_bregma, self.global_offset)
        )
        test_probe = 46116
        pts_close = True
        tst_pair = adjusted_pairs_by_probe[test_probe]
        ref_pair = self.man_calibration_pts[test_probe]
        for i in range(2):
            if not np.allclose(
                tst_pair[i],
                ref_pair[i],
                atol=1e-2,
            ):
                pts_close = False
                break
        self.assertTrue(pts_close)
        self.assertTrue(
            np.allclose(
                errs_by_probe[46116], self.man_cal_errs[46116], atol=1e-2
            )
        )

    def test_debug_parallax_calibration(self) -> None:
        """Tests for debug_parallax_calibration"""
        (
            cal_by_probe,
            R_reticle_to_bregma,
            adjusted_pairs_by_probe,
            errs_by_probe,
        ) = debug_parallax_calibration(
            self.parallax_calibration_path,
            self.global_offset,
            self.global_rotation_degrees,
        )

        self.helper_test_calibration(
            cal_by_probe, self.parallax_test_pairs, atol=1e-2
        )
        self.assertTrue(np.array_equal(R_reticle_to_bregma, np.eye(3)))
        for k, v in self.parallax_corrected_calibration_pts.items():
            reticle_pts, probe_pts = v
            received_reticle_pts, received_probe_pts = adjusted_pairs_by_probe[
                k
            ]
            self.assertTrue(
                np.allclose(reticle_pts, received_reticle_pts, atol=1e-2)
                and np.allclose(probe_pts, received_probe_pts, atol=1e-2)
            )
        # Assuming errs_by_probe is a dictionary with probe IDs as keys
        for probe_id, errs in self.par_cal_errs.items():
            self.assertTrue(
                np.allclose(errs, errs_by_probe[probe_id], atol=1e-2)
            )

    def test_read_parallel_calibration_file(self) -> None:
        """Tests for read_parallax_calibration_file"""
        mats_by_controller = read_parallax_calibration_file(
            self.parallax_example_file
        )
        reticle_pts, probe_pts = mats_by_controller[46105]
        self.assertTrue(
            np.allclose(reticle_pts, self.parallax_calibration_pts[46105][0])
            and np.allclose(probe_pts, self.parallax_calibration_pts[46105][1])
        )

    def test_read_parallax_calibration_dir(self) -> None:
        """Tests for read_parallax_calibration_dir"""
        mat_pairs_by_probe = read_parallax_calibration_dir(
            self.parallax_calibration_path
        )
        for k, v in self.parallax_calibration_pts.items():
            reticle_pts, probe_pts = v
            received_reticle_pts, received_probe_pts = mat_pairs_by_probe[k]
            self.assertTrue(
                np.allclose(reticle_pts, received_reticle_pts, atol=1e-2)
                and np.allclose(probe_pts, received_probe_pts, atol=1e-2)
            )

    def test_read_parallax_calibration_dir_and_correct(self) -> None:
        """Tests for read_parallax_calibration_dir"""
        mat_pairs_by_probe = read_parallax_calibration_dir_and_correct(
            self.parallax_calibration_path,
            self.global_offset,
            self.global_rotation_degrees,
        )
        for k, v in self.parallax_corrected_calibration_pts.items():
            reticle_pts, probe_pts = v
            received_reticle_pts, received_probe_pts = mat_pairs_by_probe[k]
            self.assertTrue(
                np.allclose(reticle_pts, received_reticle_pts, atol=1e-2)
                and np.allclose(probe_pts, received_probe_pts, atol=1e-2)
            )

    def test_fit_rotation_params_from_parallax(self) -> None:
        """Tests for fit_rotation_params_from_manual_calibration"""
        cal_by_probe, R_reticle_to_bregma = fit_rotation_params_from_parallax(
            self.parallax_calibration_path,
            self.global_offset,
            self.global_rotation_degrees,
        )
        self.helper_test_calibration(
            cal_by_probe, self.parallax_test_pairs, atol=1e-2
        )
        self.assertTrue(np.array_equal(R_reticle_to_bregma, np.eye(3)))

    def test_find_probe_insertion_vector(self) -> None:
        cal_by_probe, R_reticle_to_bregma = fit_rotation_params_from_parallax(
            self.parallax_calibration_path,
            self.global_offset,
            self.global_rotation_degrees,
        )
        R, t, s = cal_by_probe[46105]
        received_insertion_vector = find_probe_insertion_vector(R)
        self.assertTrue(
            np.allclose(
                received_insertion_vector,
                np.array([0.21, 0.28, -0.94]),
                atol=1e-2,
            )
        )

    def test_find_probe_angles(self) -> None:
        """Tests for find_probe_angle"""
        cal_by_probe, R_reticle_to_bregma = fit_rotation_params_from_parallax(
            self.parallax_calibration_path,
            self.global_offset,
            self.global_rotation_degrees,
        )
        R, t, s = cal_by_probe[46105]
        ap_angle, ml_angle = find_probe_angle(R)
        self.assertAlmostEqual(ap_angle, -16.32, places=2)
        self.assertAlmostEqual(ml_angle, -12.61, places=2)

    def test_transform_reticle_to_bregma(self) -> None:
        """Tests for transform_reticle_to_bregma"""
        reticle_pts = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        transformed_pts = transform_reticle_to_bregma(reticle_pts, R, t)
        expected_pts = reticle_pts + t
        self.assertTrue(np.allclose(transformed_pts, expected_pts))

    def test_transform_bregma_to_reticle(self) -> None:
        """Tests for transform_bregma_to_reticle"""
        bregma_pts = np.array([[2.0, 3.0, 4.0], [0.0, 1.0, 2.0]])
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        transformed_pts = transform_bregma_to_reticle(bregma_pts, R, t)
        expected_pts = bregma_pts - t
        self.assertTrue(np.allclose(transformed_pts, expected_pts))

    def test_combine_parallax_and_manual_calibrations(self) -> None:
        """Tests for combine_parallax_and_manual_calibrations"""
        cal_by_probe_combined, R_reticle_to_bregma, global_offset = (
            combine_parallax_and_manual_calibrations(
                self.manual_calibration_file,
                self.parallax_calibration_path,
            )
        )
        self.helper_test_calibration(
            cal_by_probe_combined, self.manual_test_pairs, atol=1e-2
        )
        self.helper_test_calibration(
            cal_by_probe_combined, self.parallax_test_pairs, atol=1e-2
        )
        self.assertTrue(np.array_equal(R_reticle_to_bregma, np.eye(3)))
        self.assertTrue(np.array_equal(global_offset, self.global_offset))

    def test_debug_parallax_and_manual_calibrations(self) -> None:
        """Tests for debug_parallax_and_manual_calibrations"""
        (
            combined_cal_by_probe,
            R_reticle_to_bregma,
            t_reticle_to_bregma,
            combined_pairs_by_probe,
            errs_by_probe,
        ) = debug_parallax_and_manual_calibrations(
            self.manual_calibration_file,
            self.parallax_calibration_path,
        )
        self.helper_test_calibration(
            combined_cal_by_probe, self.manual_test_pairs, atol=1e-2
        )
        self.helper_test_calibration(
            combined_cal_by_probe, self.parallax_test_pairs, atol=1e-2
        )
        self.assertTrue(np.array_equal(R_reticle_to_bregma, np.eye(3)))
        self.assertTrue(
            np.array_equal(t_reticle_to_bregma, self.global_offset)
        )
        for k, v in self.parallax_corrected_calibration_pts.items():
            reticle_pts, probe_pts = v
            received_reticle_pts, received_probe_pts = combined_pairs_by_probe[
                k
            ]
            self.assertTrue(
                np.allclose(reticle_pts, received_reticle_pts, atol=1e-2)
                and np.allclose(probe_pts, received_probe_pts, atol=1e-2)
            )
        for probe_id, errs in self.par_cal_errs.items():
            self.assertTrue(
                np.allclose(errs, errs_by_probe[probe_id], atol=1e-2)
            )


if __name__ == "__main__":
    unittest.main()
