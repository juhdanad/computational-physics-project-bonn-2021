use mcmc_algorithms::*;
use ndarray::{ArrayViewMut2, ArrayViewMut3, Dim, ShapeBuilder};
use spinconfig::*;
use std::panic::catch_unwind;
use std::slice;
use wolff_algorithms::*;

mod mcmc_algorithms;
mod spinconfig;
mod wolff_algorithms;

#[no_mangle]
pub unsafe extern "C" fn do_measure(
    w: cty::size_t,
    h: cty::size_t,
    d: cty::size_t,
    data: *mut cty::int8_t,
    m_out: *mut cty::int32_t,
    h_out: *mut cty::int32_t,
) -> bool {
    catch_unwind(|| {
        let data = std::slice::from_raw_parts_mut(data, w * h * d);
        let (m, h) = if d == 1 {
            SpinConfig2D::new(ArrayViewMut2::from_shape((h, w).strides((w, 1)), data).unwrap())
                .measure()
        } else {
            SpinConfig3D::new(
                ArrayViewMut3::from_shape((d, h, w).strides((h * w, w, 1)), data).unwrap(),
            )
            .measure()
        };
        *m_out = m;
        *h_out = h;
    })
    .is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn do_metropolis_hastings_sweep(
    w: cty::size_t,
    h: cty::size_t,
    d: cty::size_t,
    data: *mut cty::int8_t,
    beta: cty::c_double,
    j: cty::c_double,
    n: cty::int32_t,
) -> bool {
    catch_unwind(|| {
        let data = std::slice::from_raw_parts_mut(data, w * h * d);
        if d == 1 {
            let mut data =
                SpinConfig2D::new(ArrayViewMut2::from_shape((h, w).strides((w, 1)), data).unwrap());
            for _ in 0..n {
                metropolis_hastings_sweep(&mut data, beta, j);
            }
        } else {
            let mut data = SpinConfig3D::new(
                ArrayViewMut3::from_shape((d, h, w).strides((h * w, w, 1)), data).unwrap(),
            );
            for _ in 0..n {
                metropolis_hastings_sweep(&mut data, beta, j);
            }
        };
    })
    .is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn do_metropolis_hastings_measurement(
    w: cty::size_t,
    h: cty::size_t,
    d: cty::size_t,
    data: *mut cty::int8_t,
    beta: cty::c_double,
    j: cty::c_double,
    n: cty::size_t,
    accept_out: *mut cty::uint32_t,
    del_m_out: *mut cty::c_double,
    del_h_out: *mut cty::c_double,
) -> bool {
    catch_unwind(|| {
        let del_m_out = slice::from_raw_parts_mut(del_m_out, n);
        let del_h_out = slice::from_raw_parts_mut(del_h_out, n);
        let data = slice::from_raw_parts_mut(data, w * h * d);
        *accept_out = if d == 1 {
            let mut data =
                SpinConfig2D::new(ArrayViewMut2::from_shape((h, w).strides((w, 1)), data).unwrap());
            metropolis_hastings_measurement(&mut data, beta, j, del_m_out, del_h_out)
        } else {
            let mut data = SpinConfig3D::new(
                ArrayViewMut3::from_shape((d, h, w).strides((h * w, w, 1)), data).unwrap(),
            );
            metropolis_hastings_measurement(&mut data, beta, j, del_m_out, del_h_out)
        };
    })
    .is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn do_wolff_step_at(
    w: cty::size_t,
    h: cty::size_t,
    d: cty::size_t,
    data: *mut cty::int8_t,
    beta: cty::c_double,
    j: cty::c_double,
    x: cty::size_t,
    y: cty::size_t,
    z: cty::size_t,
    del_m: *mut cty::int32_t,
) -> bool {
    catch_unwind(|| {
        let data = std::slice::from_raw_parts_mut(data, w * h * d);
        *del_m = if d == 1 {
            let mut visit_stack = Vec::with_capacity(w * h * d / 2);
            wolff_step_general_at(
                &mut SpinConfig2D::new(
                    ArrayViewMut2::from_shape((h, w).strides((w, 1)), data).unwrap(),
                ),
                beta,
                j,
                &mut visit_stack,
                Dim([x, y]),
            )
        } else {
            let mut visit_stack = Vec::with_capacity(w * h * d / 2);
            wolff_step_general_at(
                &mut SpinConfig3D::new(
                    ArrayViewMut3::from_shape((d, h, w).strides((h * w, w, 1)), data).unwrap(),
                ),
                beta,
                j,
                &mut visit_stack,
                Dim([x, y, z]),
            )
        }
    })
    .is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn do_wolff_step(
    w: cty::size_t,
    h: cty::size_t,
    d: cty::size_t,
    data: *mut cty::int8_t,
    beta: cty::c_double,
    j: cty::c_double,
    n: cty::int32_t,
) -> bool {
    catch_unwind(|| {
        let data = std::slice::from_raw_parts_mut(data, w * h * d);
        if d == 1 {
            let mut visit_stack = Vec::with_capacity(w * h * d / 2);
            let mut data =
                SpinConfig2D::new(ArrayViewMut2::from_shape((h, w).strides((w, 1)), data).unwrap());
            for _ in 0..n {
                let i = data.random_index();
                wolff_step_general_at(&mut data, beta, j, &mut visit_stack, i);
            }
        } else {
            let mut visit_stack = Vec::with_capacity(w * h * d / 2);
            let mut data = SpinConfig3D::new(
                ArrayViewMut3::from_shape((d, h, w).strides((h * w, w, 1)), data).unwrap(),
            );
            for _ in 0..n {
                let i = data.random_index();
                wolff_step_general_at(&mut data, beta, j, &mut visit_stack, i);
            }
        }
    })
    .is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn do_wolff_measurement(
    w: cty::size_t,
    h: cty::size_t,
    d: cty::size_t,
    data: *mut cty::int8_t,
    beta: cty::c_double,
    j: cty::c_double,
    n: cty::size_t,
    cluster_size_out: *mut cty::c_double,
    m_out: *mut cty::c_double,
) -> bool {
    catch_unwind(|| {
        let data = std::slice::from_raw_parts_mut(data, w * h * d);
        let m_out = slice::from_raw_parts_mut(m_out, n);
        let cluster_size_out = slice::from_raw_parts_mut(cluster_size_out, n);
        if d == 1 {
            let mut data =
                SpinConfig2D::new(ArrayViewMut2::from_shape((h, w).strides((w, 1)), data).unwrap());
            wolff_measurement(&mut data, beta, j, cluster_size_out, m_out);
        } else {
            let mut data = SpinConfig3D::new(
                ArrayViewMut3::from_shape((d, h, w).strides((h * w, w, 1)), data).unwrap(),
            );
            wolff_measurement(&mut data, beta, j, cluster_size_out, m_out);
        }
    })
    .is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn do_wolff_measurement_full(
    w: cty::size_t,
    h: cty::size_t,
    d: cty::size_t,
    data: *mut cty::int8_t,
    beta: cty::c_double,
    j: cty::c_double,
    n: cty::size_t,
    cluster_size_out: *mut cty::c_double,
    m_out: *mut cty::c_double,
    e_out: *mut cty::c_double,
) -> bool {
    catch_unwind(|| {
        let data = std::slice::from_raw_parts_mut(data, w * h * d);
        let m_out = slice::from_raw_parts_mut(m_out, n);
        let cluster_size_out = slice::from_raw_parts_mut(cluster_size_out, n);
        let e_out = slice::from_raw_parts_mut(e_out, n);
        if d == 1 {
            let mut data =
                SpinConfig2D::new(ArrayViewMut2::from_shape((h, w).strides((w, 1)), data).unwrap());
            wolff_measurement_full(&mut data, beta, j, cluster_size_out, m_out, e_out);
        } else {
            let mut data = SpinConfig3D::new(
                ArrayViewMut3::from_shape((d, h, w).strides((h * w, w, 1)), data).unwrap(),
            );
            wolff_measurement_full(&mut data, beta, j, cluster_size_out, m_out, e_out);
        }
    })
    .is_ok()
}
