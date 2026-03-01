import h5py
import hdf5plugin  # noqa: F401


def _load_events_h5(path: str, num_events: int) -> dict:
    with h5py.File(path, "r") as f:
        events = f["events"]
        t = events["t"][-num_events:]
        x = events["x"][-num_events:]
        y = events["y"][-num_events:]
        p = events["p"][-num_events:]
    return dict(x=x, y=y, t=t, p=p)