use ndarray::Dimension;
use rand::random;

use crate::spinconfig::SpinConfig;

pub fn wolff_step_general_at<'a, D, I>(
    data: &mut D,
    beta: f64,
    j: f64,
    visit_stack: &mut Vec<I>,
    pos: I,
) -> i32
where
    D: SpinConfig<'a, I>,
    I: Dimension + Copy,
{
    let threshold = (-2.0 * beta * j).exp();
    // get random coordinates
    visit_stack.clear();
    data.spins()[pos] *= -1;
    visit_stack.push(pos);
    let mut change = data.spins()[pos] as i32;
    while let Some(current_point) = visit_stack.pop() {
        let current_val = data.spins()[current_point];
        data.for_all_neighbors(current_point, |data, neighbor| {
            let neighbor_val = data.spins()[neighbor];
            if neighbor_val != current_val && random::<f64>() > threshold {
                data.spins()[neighbor] *= -1;
                change -= neighbor_val as i32;
                visit_stack.push(neighbor);
            }
        });
    }
    return change * 2;
}

pub fn wolff_measurement<'a, D, I>(
    data: &mut D,
    beta: f64,
    j: f64,
    cluster_size_out: &mut [f64],
    m_out: &mut [f64],
) where
    D: SpinConfig<'a, I>,
    I: Dimension + Copy,
{
    let m_out = m_out.iter_mut();
    let cluster_size_out = cluster_size_out.iter_mut();
    let mut visit_stack = Vec::with_capacity(data.spins().len() / 2);
    let (mut m0, _) = data.measure();
    for (m, c) in m_out.zip(cluster_size_out) {
        let i = data.random_index();
        let m1 = wolff_step_general_at(data, beta, j, &mut visit_stack, i);
        m0 += m1;
        *m = m0 as f64;
        *c = (m1 / 2).abs() as f64;
    }
}

pub fn wolff_measurement_full<'a, D, I>(
    data: &mut D,
    beta: f64,
    j: f64,
    cluster_size_out: &mut [f64],
    m_out: &mut [f64],
    e_out: &mut [f64],
) where
    D: SpinConfig<'a, I>,
    I: Dimension + Copy,
{
    let m_out = m_out.iter_mut();
    let e_out = e_out.iter_mut();
    let cluster_size_out = cluster_size_out.iter_mut();
    let mut visit_stack = Vec::with_capacity(data.spins().len() / 2);
    for ((m, c), e) in m_out.zip(cluster_size_out).zip(e_out) {
        let i = data.random_index();
        *c = (wolff_step_general_at(data, beta, j, &mut visit_stack, i) / 2).abs() as f64;
        let (m1, e1) = data.measure();
        *m = m1 as f64;
        *e = e1 as f64 * j;
    }
}
