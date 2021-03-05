use ndarray::prelude::*;
use rand::{distributions::Uniform, prelude::*};

pub trait SpinConfig<'a, I: Dimension + Copy> {
    fn random_index(&mut self) -> I;
    fn for_all_neighbors<F>(&mut self, index: I, f: F)
    where
        F: FnMut(&mut Self, I);
    fn spins(&mut self) -> &mut ArrayViewMut<'a, i8, I>;
    /// Returns the magnetic moment and the energy (with coupling J=1)
    fn measure(&mut self) -> (i32, i32);
}

pub struct SpinConfig2D<'a> {
    /// Spin configuration. A square array, where each element is 1 or -1.
    config: ArrayViewMut2<'a, i8>,
    /// Uniformly samples integers from [0,N), where N is the size
    /// (width or height) of the configuration.
    index_distribution: [Uniform<usize>; 2],
    rng: ThreadRng,
}

impl SpinConfig2D<'_> {
    pub fn new(arr: ArrayViewMut2<i8>) -> SpinConfig2D {
        SpinConfig2D {
            index_distribution: [
                Uniform::from(0..arr.shape()[0]),
                Uniform::from(0..arr.shape()[1]),
            ],
            config: arr,
            rng: thread_rng(),
        }
    }
}

impl<'a> SpinConfig<'a, Ix2> for SpinConfig2D<'a> {
    fn random_index(&mut self) -> Ix2 {
        Dim([
            self.rng.sample(self.index_distribution[0]),
            self.rng.sample(self.index_distribution[1]),
        ])
    }

    fn for_all_neighbors<F>(&mut self, index: Ix2, mut f: F)
    where
        F: FnMut(&mut Self, Ix2),
    {
        let w = self.config.shape()[0];
        let h = self.config.shape()[1];
        f(self, Dim([index[0], (index[1] + 1) % h]));
        f(
            self,
            Dim([index[0], index[1].checked_sub(1).unwrap_or(h - 1)]),
        );
        f(self, Dim([(index[0] + 1) % w, index[1]]));
        f(
            self,
            Dim([index[0].checked_sub(1).unwrap_or(w - 1), index[1]]),
        );
    }

    fn spins(&mut self) -> &mut ArrayViewMut<'a, i8, Ix2> {
        &mut self.config
    }

    fn measure(&mut self) -> (i32, i32) {
        let config = &self.config;
        let h = config.nrows();
        let w = config.ncols();
        let mut m = 0;
        let mut e = 0;
        // for each spin
        for r in 0..h {
            for c in 0..w {
                m += config[[r, c]] as i32;
                e -=
                    (config[[r, c]] * (config[[r, (c + 1) % w]] + config[[(r + 1) % h, c]])) as i32;
            }
        }
        (m, e)
    }
}

pub struct SpinConfig3D<'a> {
    /// Spin configuration. A square array, where each element is 1 or -1.
    config: ArrayViewMut3<'a, i8>,
    /// Uniformly samples integers from [0,N), where N is the size
    /// (width or height) of the configuration.
    index_distribution: [Uniform<usize>; 3],
    rng: ThreadRng,
}

impl SpinConfig3D<'_> {
    pub fn new(arr: ArrayViewMut3<i8>) -> SpinConfig3D {
        SpinConfig3D {
            index_distribution: [
                Uniform::from(0..arr.shape()[0]),
                Uniform::from(0..arr.shape()[1]),
                Uniform::from(0..arr.shape()[2]),
            ],
            config: arr,
            rng: thread_rng(),
        }
    }
}

impl<'a> SpinConfig<'a, Ix3> for SpinConfig3D<'a> {
    fn random_index(&mut self) -> Ix3 {
        Dim([
            self.rng.sample(self.index_distribution[0]),
            self.rng.sample(self.index_distribution[1]),
            self.rng.sample(self.index_distribution[2]),
        ])
    }

    fn for_all_neighbors<F>(&mut self, index: Ix3, mut f: F)
    where
        F: FnMut(&mut Self, Ix3),
    {
        let w = self.config.shape()[0];
        let h = self.config.shape()[1];
        let d = self.config.shape()[2];
        f(self, Dim([index[0], (index[1] + 1) % h, index[2]]));
        f(
            self,
            Dim([index[0], index[1].checked_sub(1).unwrap_or(h - 1), index[2]]),
        );
        f(self, Dim([(index[0] + 1) % w, index[1], index[2]]));
        f(
            self,
            Dim([index[0].checked_sub(1).unwrap_or(w - 1), index[1], index[2]]),
        );
        f(self, Dim([index[0], index[1], (index[2] + 1) % d]));
        f(
            self,
            Dim([index[0], index[1], index[2].checked_sub(1).unwrap_or(d - 1)]),
        );
    }

    fn spins(&mut self) -> &mut ArrayViewMut<'a, i8, Ix3> {
        &mut self.config
    }

    fn measure(&mut self) -> (i32, i32) {
        let config = &self.config;
        let (x_max, y_max, z_max) = config.dim();
        let mut m = 0;
        let mut e = 0;
        // for each spin
        for x in 0..x_max {
            for y in 0..y_max {
                for z in 0..z_max {
                    m += config[[x, y, z]] as i32;
                    e -= (config[[x, y, z]]
                        * (config[[x, y, (z + 1) % z_max]]
                            + config[[x, (y + 1) % y_max, z]]
                            + config[[(x + 1) % x_max, y, z]])) as i32;
                }
            }
        }
        (m, e)
    }
}
