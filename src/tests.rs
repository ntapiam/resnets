#[cfg(test)]
mod tests {
    use crate::p_var::*;
    use rand::distributions::Standard;
    use rand::prelude::*;

    fn compute_dist<'z, const N: usize>(v: &'z [[f64; N]]) -> Vec<Vec<f64>> {
        v.iter()
            .map(|&a| {
                v.iter()
                    .map(|&b| {
                        a.iter()
                            .zip(b)
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn square_path() {
        let v = [
            [0., 0.],
            [0., 1.],
            [1., 1.],
            [1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ];
        let mut p = 1.;
        let dist = compute_dist(&v);
        while p <= 4. {
            assert_eq!(
                p_var_backbone(&v, p, &dist).ok(),
                p_var_backbone_ref(&v, p, &dist).ok()
            );
            p += 0.5;
        }
    }

    #[test]
    fn bm() {
        const N: usize = 2500;
        let mut path = [0.; N + 1];
        let sigma = 1. / f64::sqrt(N as f64);
        path[1..].copy_from_slice(
            &StdRng::from_entropy()
                .sample::<[bool; N], Standard>(Standard)
                .map(|x| if x { sigma } else { -sigma })
                .iter()
                .scan(0., |acc, x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect::<Vec<_>>(),
        );
        let mut path2 = vec![];
        path.iter().for_each(|&x| path2.push([x]));
        let dist = compute_dist(&path2);
        for p in [1., f64::sqrt(2.), 2., f64::exp(1.)] {
            assert_eq!(
                p_var_backbone(&path, p, &dist),
                p_var_backbone_ref(&path, p, &dist)
            );
        }
    }
}
