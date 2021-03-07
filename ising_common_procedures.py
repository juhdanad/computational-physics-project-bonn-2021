import numpy as np
import ctypes as ct
from numpy.ctypeslib import ndpointer

beta_c = beta_c=np.log(1+np.sqrt(2))/2
beta_c_3d = 0.2216544

def init_grid(size):
    return np.random.choice(np.array([-1,1],dtype=np.int8),size)

def flip_grid_element(config):
    x = np.random.randint(0, len(config))
    y = np.random.randint(0, len(config))

    new_config = config.copy()
    new_config[x][y] *= -1

    return new_config

compiled_c_code = ct.CDLL("./final_project_native/target/release/libfinal_project_native.so")

ret32_1 = ct.c_int32()
ret32_2 = ct.c_int32()
retu32_1 = ct.c_uint32()

compiled_c_code.do_measure.argtypes = [ct.c_size_t,ct.c_size_t, ct.c_size_t,\
                        ndpointer(dtype=np.int8,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
compiled_c_code.do_measure.restype = ct.c_bool
# Returns the magnetic moment and energy for the given configuration
def measure(arr,j):
    if arr.ndim==2:
        assert compiled_c_code.do_measure(arr.shape[1],arr.shape[0],1,arr,ret32_1,ret32_2), "Rust panic! See cmd line."
    else:
        assert compiled_c_code.do_measure(arr.shape[2],arr.shape[1],arr.shape[0],arr,ret32_1,ret32_2), "Rust panic! See cmd line."
    return ret32_1.value, ret32_2.value*j

compiled_c_code.do_metropolis_hastings_sweep.argtypes = [ct.c_size_t,ct.c_size_t, ct.c_size_t,\
                        ndpointer(dtype=np.int8,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ct.c_double, ct.c_double, ct.c_int32]
compiled_c_code.do_metropolis_hastings_sweep.restype = ct.c_bool
# Executes n Metropolis-Hastings sweeps.
def metropolis_hastings_sweep(arr,beta,j,n=1):
    if arr.ndim==2:
        assert compiled_c_code.do_metropolis_hastings_sweep(arr.shape[1],arr.shape[0],1,arr,beta,j,n), "Rust panic! See cmd line."
    else:
        assert compiled_c_code.do_metropolis_hastings_sweep(arr.shape[2],arr.shape[1],arr.shape[0],arr,beta,j,n), "Rust panic! See cmd line."

compiled_c_code.do_metropolis_hastings_measurement.argtypes = [ct.c_size_t,ct.c_size_t, ct.c_size_t,\
                        ndpointer(dtype=np.int8,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ct.c_double, ct.c_double, ct.c_int32, ct.POINTER(ct.c_uint32),\
                        ndpointer(dtype=np.float64,ndim=1,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ndpointer(dtype=np.float64,ndim=1,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE'))]
compiled_c_code.do_metropolis_hastings_measurement.restype = ct.c_bool
# Executes as many Metropolis-Hastings sweeps as the length of the output arrays.
# Returns the total number of accepted flips.
# Fills the m_out and e_out arrays with the magnetic moment and energy values measured after each sweep.
def metropolis_hastings_measurement(config,beta,j,m_out,e_out):
    assert m_out.size==e_out.size
    if config.ndim==2:
        assert compiled_c_code.do_metropolis_hastings_measurement(config.shape[1],config.shape[0],1,config,beta,j,m_out.size,retu32_1,m_out,e_out), "Rust panic! See cmd line."
    else:
        assert compiled_c_code.do_metropolis_hastings_measurement(config.shape[2],config.shape[1],config.shape[0],config,beta,j,m_out.size,retu32_1,m_out,e_out), "Rust panic! See cmd line."
    return retu32_1.value

compiled_c_code.do_wolff_step_at.argtypes = [ct.c_size_t,ct.c_size_t, ct.c_size_t,\
                        ndpointer(dtype=np.int8,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ct.c_double, ct.c_double, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.POINTER(ct.c_int32)]
compiled_c_code.do_wolff_step_at.restype = ct.c_bool
# Executes a Wolff step on the array, with coupling j, and starting from (x,y,z)
# Requires a 2D/3D Numpy array of dtype=numpy.int8, each element +1 or -1.
# Returns the number of changed spins * the new sign of the cluster.
def wolff_step_at(arr,beta,j,x,y,z=0):
    if arr.ndim==2:
        assert compiled_c_code.do_wolff_step_at(arr.shape[1],arr.shape[0],1,arr,beta, j,x,y,z,ret32_1), "Rust panic! See cmd line."
    else:
        assert compiled_c_code.do_wolff_step_at(arr.shape[2],arr.shape[1],arr.shape[0],arr,beta,j,x,y,z,ret32_1), "Rust panic! See cmd line."
    return ret32_1.value

compiled_c_code.do_wolff_step.argtypes = [ct.c_size_t,ct.c_size_t, ct.c_size_t,\
                        ndpointer(dtype=np.int8,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ct.c_double, ct.c_double, ct.c_int32]
compiled_c_code.do_wolff_step.restype = ct.c_bool
# Executes n Wolff steps on the array, with coupling j.
def wolff_step(arr,beta, j,n=1):
    if arr.ndim==2:
        assert compiled_c_code.do_wolff_step(arr.shape[1],arr.shape[0],1,arr,beta,j,n), "Rust panic! See cmd line."
    else:
        assert compiled_c_code.do_wolff_step(arr.shape[2],arr.shape[1],arr.shape[0],arr,beta,j,n), "Rust panic! See cmd line."
        
compiled_c_code.do_wolff_measurement.argtypes = [ct.c_size_t,ct.c_size_t, ct.c_size_t,\
                        ndpointer(dtype=np.int8,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ct.c_double, ct.c_double, ct.c_int32,\
                        ndpointer(dtype=np.float64,ndim=1,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ndpointer(dtype=np.float64,ndim=1,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE'))]
compiled_c_code.do_wolff_measurement.restype = ct.c_bool
# Executes as many Wolff steps on the array as the length of the output arrays.
# Fills the cluster_size_out and m_out arrays with the latest cluster size and
# magnetic moment measured after each sweep.
def wolff_measurement(config,beta,j,cluster_size_out,m_out):
    assert m_out.size==cluster_size_out.size
    if config.ndim==2:
        assert compiled_c_code.do_wolff_measurement(config.shape[1],config.shape[0],1,config,beta,j,m_out.size,cluster_size_out,m_out), "Rust panic! See cmd line."
    else:
        assert compiled_c_code.do_wolff_measurement(config.shape[2],config.shape[1],config.shape[0],config,beta,j,m_out.size,cluster_size_out,m_out), "Rust panic! See cmd line."
        
compiled_c_code.do_wolff_measurement_full.argtypes = [ct.c_size_t,ct.c_size_t, ct.c_size_t,\
                        ndpointer(dtype=np.int8,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ct.c_double, ct.c_double, ct.c_int32,\
                        ndpointer(dtype=np.float64,ndim=1,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ndpointer(dtype=np.float64,ndim=1,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE')),\
                        ndpointer(dtype=np.float64,ndim=1,flags=('C_CONTIGUOUS','ALIGNED','WRITEABLE'))]
compiled_c_code.do_wolff_measurement_full.restype = ct.c_bool
# Executes as many Wolff steps on the array as the length of the output arrays.
# Fills the cluster_size_out, m_out and e_out arrays with the latest cluster size,
# magnetic moment and energy measured after each sweep.
def wolff_measurement_full(config,beta,j,cluster_size_out,m_out,e_out):
    assert m_out.size==cluster_size_out.size
    assert e_out.size==cluster_size_out.size
    if config.ndim==2:
        assert compiled_c_code.do_wolff_measurement_full(config.shape[1],config.shape[0],1,config,beta,j,m_out.size,cluster_size_out,m_out,e_out), "Rust panic! See cmd line."
    else:
        assert compiled_c_code.do_wolff_measurement_full(config.shape[2],config.shape[1],config.shape[0],config,beta,j,m_out.size,cluster_size_out,m_out,e_out), "Rust panic! See cmd line."
