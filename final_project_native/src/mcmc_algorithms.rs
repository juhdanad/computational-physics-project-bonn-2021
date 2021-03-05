use ndarray::Dimension;
use rand::random;

use crate::spinconfig::SpinConfig;

/// Performs as many random accept/reject steps, as many spins are in the lattice.
/// Returns: (#accepted, delta M, delta H), with (J=1)
pub fn metropolis_hastings_sweep<'a, D, I>(data: &mut D, beta: f64, j: f64) -> (u32, i32, i32)
where
    D: SpinConfig<'a, I>,
    I: Dimension + Copy,
{
    let (mut num_accept, mut delta_m, mut delta_h) = (0, 0, 0);
    // repeat as many times as the number of spins
    for _ in 0..data.spins().len() {
        // get random coordinates
        let index = data.random_index();
        // get change for the proposed flip
        let mut sum_of_neighbors: i8 = 0;
        data.for_all_neighbors(index, |data, neighbor| {
            sum_of_neighbors += data.spins()[neighbor];
        });
        let delta_h_curr = 2 * sum_of_neighbors * data.spins()[index];
        if random::<f64>() < (-beta * j * delta_h_curr as f64).exp() {
            // actually do the flip if accepted
            data.spins()[index] *= -1;
            delta_h += delta_h_curr as i32;
            num_accept += 1;
            delta_m += data.spins()[index] as i32;
        }
    }
    (num_accept, delta_m * 2, delta_h)
}

pub fn metropolis_hastings_measurement<'a, D, I>(
    data: &mut D,
    beta: f64,
    j: f64,
    m_out: &mut [f64],
    h_out: &mut [f64],
) -> u32
where
    D: SpinConfig<'a, I>,
    I: Dimension + Copy,
{
    let m_iter = m_out.iter_mut();
    let h_iter = h_out.iter_mut();
    let mut a0 = 0;
    let (mut m0, mut h0) = data.measure();
    for (m, h) in m_iter.zip(h_iter) {
        let (a1, m1, h1) = metropolis_hastings_sweep(data, beta, j);
        a0 += a1;
        m0 += m1;
        h0 += h1;
        *m = m0 as f64;
        *h = h0 as f64 * j;
    }
    a0
}
