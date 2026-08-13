# Urban4--Topotherm heat-topology comparison

This controlled benchmark compares only the source-to-customer heat routes.
It keeps the Urban4 H1/H2 sources, selected customers, source assignment,
customer peaks, and OpenStreetMap road data fixed. Building services and the
H1--H2 intertie are excluded from both route totals.

Topotherm 0.6.0 is run in forced-connection, single-timestep mode with HiGHS.
The candidate graph is cropped and reduced for tractability, but it retains the
full Urban4 route. Therefore Urban4 is a feasible alternative on the graph
given to Topotherm.

| Territory | Customers | Urban4 route | Topotherm route | Difference | Status |
|---|---:|---:|---:|---:|---|
| H1 | 600 | 21.84 km | 15.06 km | -31.0% | Time limit; 9.24% gap |
| H2 | 241 | 11.30 km | 7.75 km | -31.5% | Within 1% gap |

This is a cross-tool benchmark, not validation against installed pipes. It
does not compare customer selection, pipe sizing, plant dispatch, electricity,
water, wastewater, or cross-sector coupling. The H1 Topotherm result is a
feasible time-limited solution, not a proven optimum.
