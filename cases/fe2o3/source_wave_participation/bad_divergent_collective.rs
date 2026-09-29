// Source-level analogue of quant r002/r006's divergent HIP wave shuffle.
use fe2o3_device::{kernel, thread, Gfx942Collectives};

#[kernel(
    typed,
    namespace = "0c181b24f360a4b30f4f79e64cf579273d2239bbcdfdfea06003f40e82de7d53"
)]
pub fn gfx942_wave_lds_v1(active_flag: u32, value: u32) {
    let context = Gfx942Collectives::current();
    let lane = thread::index_1d().get();
    let wave_sum = if lane & 1 == 0 {
        context.wave64_reduce_sum_active_u32(1, value)
    } else {
        0
    };
    let mut scratch = context.static_lds_u32x256();
    let workgroup_sum =
        context.workgroup256_reduce_sum_active_u32(&mut scratch, active_flag, value);
    let _ = (wave_sum, workgroup_sum);
}

fn main() {}
