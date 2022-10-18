#https://en.wikipedia.org/wiki/Pseudorandom_number_generator#Implementation
pseudo_randomf = lambda seed: ((1234*(seed+1235))%1000)/1000#((((seed*seed*seed+1)*7901) % 6997)) / 6997
pseudo_randomf_minus1_1 = lambda seed: 2*pseudo_randomf(seed) - 1