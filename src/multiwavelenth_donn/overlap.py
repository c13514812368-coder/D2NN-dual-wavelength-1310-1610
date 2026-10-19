import torch
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
from scipy.interpolate import PchipInterpolator
import numpy as np
from scipy.io import savemat, loadmat
from scipy.signal import resample as fft_resample
def extract_EH_1d_line(
    F,
    *,
    prefer_axis: str = "y",
    forbid_interp: bool = True,
    mode_index: int | None = None,
    return_axis: bool = False
):
    x = np.asarray(getattr(F, "x")).squeeze()
    y = np.asarray(getattr(F, "y")).squeeze()
    z = np.asarray(getattr(F, "z")).squeeze()

    sizes = {"x": np.size(x), "y": np.size(y), "z": np.size(z)}
    if prefer_axis.lower() == "auto":
        axis = next((a for a in ("y", "x", "z") if sizes[a] > 1), None)
        if axis is None:
            raise ValueError("Invalid input")
    else:
        axis = prefer_axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError("Invalid input")
        if sizes[axis] <= 1:
            raise ValueError("Invalid input")

    coord = {"x": x, "y": y, "z": z}[axis].astype(float)
    N = coord.size

    E = getattr(F, "E", None)
    H = getattr(F, "H", None)

    def _to_N3_inner(arr, N_expected, mode_index, name):
        if arr is None:
            return None
        a = np.asarray(arr)
        if np.iscomplexobj(a) is False:
            a = a.astype(np.complex128, copy=False)
        if a.ndim == 2 and a.shape == (N_expected, 3):
            return a
        if a.ndim == 2 and a.shape == (3, N_expected):
            return a.T
        if a.ndim == 3 and a.shape[0] == N_expected and 3 in a.shape[1:]:
            comp_axis = 1 if a.shape[1] == 3 else 2
            mode_axis = 2 if comp_axis == 1 else 1
            m = 0 if mode_index is None else mode_index
            a = np.take(a, m, axis=mode_axis)
            if comp_axis != 1:
                a = np.moveaxis(a, -1, 1)
            return a
        if a.ndim == 3 and a.shape[1] == N_expected and 3 in (a.shape[0], a.shape[2]):
            if a.shape[0] == 3:
                a = np.moveaxis(a, 0, -1)
                m = 0 if mode_index is None else mode_index
                a = a[:, m, :]
                return a
            else:
                m = 0 if mode_index is None else mode_index
                a = a[m, :, :]
                return a
        for comp_axis in range(a.ndim):
            if a.shape[comp_axis] == 3:
                other_axes = [i for i in range(a.ndim) if i != comp_axis]
                b = np.moveaxis(a, comp_axis, -1)
                b = b.reshape(-1, 3)
                if b.shape[0] != N_expected:
                    raise ValueError("Invalid input")
                return b
        raise ValueError("Invalid input")

    E = _to_N3_inner(E, N, mode_index, "E") if E is not None else None
    H = _to_N3_inner(H, N, mode_index, "H") if H is not None else None

    order = np.argsort(coord)
    if not np.all(order == np.arange(N)):
        coord = coord[order]
        if E is not None: E = E[order, :]
        if H is not None: H = H[order, :]

    if N > 1:
        d = np.diff(coord)
        dy = float(np.median(d))
        is_uniform = np.allclose(d, dy, rtol=1e-6, atol=1e-12)
        if not is_uniform:
            if forbid_interp:
                raise ValueError("Invalid input")
            coord_new = np.linspace(coord[0], coord[-1], N)
            def _interp_cols(A):
                if A is None: return None
                out = np.empty_like(A, dtype=np.complex128)
                for i in range(3):
                    out[:, i] = np.interp(coord_new, coord, A[:, i].real) \
                                + 1j*np.interp(coord_new, coord, A[:, i].imag)
                return out
            E = _interp_cols(E)
            H = _interp_cols(H)
            coord = coord_new

    def _split_cols(A):
        if A is None: return (None, None, None)
        A = A.astype(np.complex128, copy=False)
        return A[:, 0], A[:, 1], A[:, 2]

    Ex, Ey, Ez = _split_cols(E)
    Hx, Hy, Hz = _split_cols(H)

    if return_axis:
        return coord.astype(float), Ex, Ey, Ez, Hx, Hy, Hz, axis
    return coord.astype(float), Ex, Ey, Ez, Hx, Hy, Hz


from typing import Optional

def _ensure_y_for_Ey(Ey, y=None, *, span_m=12e-6, origin='left', device=None, dtype=None):
    """
    ?????? y???????? span_m/origin ?????????????:
      - origin='left'   -> [0, L]
      - origin='center' -> [-L/2, L/2]
    ??? torch ?????? numpy
    """
    dev = device if device is not None else getattr(Ey, "device", torch.device("cpu"))
    Ey_t = torch.as_tensor(Ey, dtype=torch.complex64 if dtype is None else dtype, device=dev).squeeze()
    if y is None:
        N = Ey_t.numel()
        if N < 2:
            raise ValueError("Ey ??????? 2 ????????")
        if origin == 'left':
            y_new = torch.linspace(0.0, float(span_m), N, device=dev, dtype=torch.float64)
        elif origin == 'center':
            y_new = torch.linspace(-float(span_m)/2.0, float(span_m)/2.0, N, device=dev, dtype=torch.float64)
        else:
            raise ValueError("origin ???? 'left' ?? 'center'.")
        return y_new, Ey_t

    y_t = torch.as_tensor(y, dtype=torch.float64, device=dev).squeeze()
    if y_t.ndim != 1 or y_t.numel() != Ey_t.numel():
        raise ValueError("???? y ????? 1D ??????? Ey ?????")
    if y_t.numel() < 2:
        raise ValueError("??????? 2 ????????")
    dy = float((y_t[1]-y_t[0]).item())
    if not torch.allclose(torch.diff(y_t), torch.full_like(torch.diff(y_t), dy), rtol=1e-6, atol=1e-12):
        raise ValueError("y ???????")
    return y_t, Ey_t

def _to_complex_tensor(x, *, device=None, dtype=torch.complex128):
    """
    Like torch.as_tensor but keeps gradient/device when x is already a tensor.
    """
    if isinstance(x, torch.Tensor):
        t = x if device is None else x.to(device=device)
        if not torch.is_complex(t):
            base_dtype = torch.get_default_dtype() if t.dtype.is_floating_point else torch.float32
            t = t.to(dtype=base_dtype)
            t = t.to(dtype)
        elif t.dtype != dtype:
            t = t.to(dtype)
    else:
        t = torch.tensor(x, dtype=dtype, device=device)
    return t


def _to_1d_tensor(x, dtype, *, device=None):
    if callable(x):
        raise TypeError("Invalid input type")
    if isinstance(x, torch.Tensor):
        t = x if device is None else x.to(device=device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
    else:
        t = torch.tensor(x, dtype=dtype, device=device)
    return t.squeeze()
class Overlap():
    def __init__(self,y_m,Ey_m,modepath=None, 
        Hz_m=None,
        *,
        #                       
        Ex_m=None, Ez_m=None, Hx_m=None, Hy_m=None,
        out_dir=None,
        tag="P_right",
        mode_index=1,
        wavelength_m=None,
        origin="left",
        oned_as="row",       # 'row'   'column'    1D    1xN    Nx1
        do_compression=True ,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,):
        #       torch tensor
        
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        if modepath is not None:
            self.mode_path = Path(modepath)
            self.y_m, self.Ex_m, self.Ey_m, self.Ez_m, self.Hx_m, self.Hy_m, self.Hz_m, self.zeta_m, self.meta = self.load_emode_from_mat(self.mode_path) 
        else :
            mode_path = self.save_emode_to_mat(
                            y_m=y_m,
                            Ex_m=Ex_m,
                            Ey_m=Ey_m,
                            Ez_m=Ez_m,
                            Hx_m=Hx_m,
                            Hy_m=Hy_m,
                            Hz_m=Hz_m,
                            out_dir=out_dir,    #        
                            tag=tag,
                            mode_index=mode_index,
                            wavelength_m=wavelength_m,
                            origin="left")              #   "center"        
            self.mode_path = Path(mode_path)
            self.y_m, self.Ex_m, self.Ey_m, self.Ez_m, self.Hx_m, self.Hy_m, self.Hz_m, self.zeta_m, self.meta = self.load_emode_from_mat(self.mode_path)

 
    def _as_1d(self, a, dtype=None):
        if a is None:
            return None
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        x = np.asarray(a, dtype=dtype).squeeze()
        if x.ndim != 1:
            raise ValueError("Invalid input")
        return x
    def _is_uniform(self,y, rtol=1e-7, atol=0.0):
        y = np.asarray(y, float)
        d = np.diff(y)
        return np.all(d > 0) and np.allclose(d, d[0], rtol=rtol, atol=atol)
    def _interp_complex(self,y_src, f_src, y_tgt):
    #          /     
        return np.interp(y_tgt, y_src, np.real(f_src)) + 1j*np.interp(y_tgt, y_src, np.imag(f_src))
    def save_emode_to_mat(
            self,
        y_m,
        Ey_m,
        Hz_m=None,
        *,
        #                       
        Ex_m=None, Ez_m=None, Hx_m=None, Hy_m=None,
        out_dir=None,
        tag="P_right",
        mode_index=1,
        wavelength_m=None,
        origin="left",
        oned_as="row",       # 'row'   'column'    1D    1xN    Nx1
        do_compression=True  #         
    ):
        """
           Emode   .mat     Ex/Ey/Ez/Hx/Hy/Hz  
        -        Ey/Hz         Hz     Ey         Hz   zeta 
        -        y_m, Ex_m, Ey_m, Ez_m, Hx_m, Hy_m, Hz_m, zeta_m(   )
        """
        # ---       ---
        y_m  = self._as_1d(y_m, float)
        Ey_m = self._as_1d(Ey_m, np.complex128)
        if y_m is None or Ey_m is None:
            raise ValueError("Invalid input")
        N = y_m.size
        if N < 2:
            raise ValueError("Invalid input")

        #              
        Ex_m = self._as_1d(Ex_m, np.complex128) if Ex_m is not None else None
        Ez_m = self._as_1d(Ez_m, np.complex128) if Ez_m is not None else None
        Hx_m = self._as_1d(Hx_m, np.complex128) if Hx_m is not None else None
        Hy_m = self._as_1d(Hy_m, np.complex128) if Hy_m is not None else None
        Hz_m = self._as_1d(Hz_m, np.complex128) if Hz_m is not None else None

        def _check_len(name, arr):
            if arr is not None and arr.size != N:
                raise ValueError("Invalid input")

        for nm, arr in [("Ex_m", Ex_m), ("Ey_m", Ey_m), ("Ez_m", Ez_m),
                        ("Hx_m", Hx_m), ("Hy_m", Hy_m), ("Hz_m", Hz_m)]:
            _check_len(nm, arr)

        # ---          ---
        if out_dir is None:
            try:
                out_dir = Path(sim.output_dir)   #      SimuData
            except NameError:
                out_dir = Path.cwd()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        wl_nm = int(round(float(wavelength_m) * 1e9)) if (wavelength_m is not None) else None
        base = "Emode_yExEyEzHxHyHz"   #             
        fname = f"{base}_{tag}_m{int(mode_index)}" + (f"_wl{wl_nm}nm.mat" if wl_nm is not None else ".mat")
        path = out_dir / fname

        # ---        ---
        payload = {
            "y_m": y_m,
            "Ey_m": Ey_m,  #         
            "meta": {
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "origin": str(origin),
                "y_start": float(y_m[0]),
                "y_end": float(y_m[-1]),
                "span_m": float(y_m[-1] - y_m[0]),
                "dy": (float(y_m[1] - y_m[0]) if N >= 2 else None),
                "tag": str(tag),
                "mode_index": int(mode_index),
                "wavelength_m": (float(wavelength_m) if wavelength_m is not None else None),
                "fields_present": {
                    "Ex_m": Ex_m is not None,
                    "Ey_m": True,
                    "Ez_m": Ez_m is not None,
                    "Hx_m": Hx_m is not None,
                    "Hy_m": Hy_m is not None,
                    "Hz_m": Hz_m is not None,
                },
                "oned_as": str(oned_as),
            }
        }

        #         
        if Ex_m is not None: payload["Ex_m"] = Ex_m
        if Ez_m is not None: payload["Ez_m"] = Ez_m
        if Hx_m is not None: payload["Hx_m"] = Hx_m
        if Hy_m is not None: payload["Hy_m"] = Hy_m
        if Hz_m is not None:
            payload["Hz_m"] = Hz_m
            #          zeta_m = Ey/Hz
            zeta_m = Ey_m / (Hz_m + 1e-30)
            payload["zeta_m"] = zeta_m

        #    .mat oned_as        /            
        savemat(str(path), payload, oned_as=oned_as, do_compression=do_compression)
        return path
    def load_emode_from_mat(self,mat_path):
        """
           .mat    (y_m, Ex_m, Ey_m, Ez_m, Hx_m, Hy_m, Hz_m, zeta_m, meta)
        -          Ey     Hz/zeta         
        """
        D = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        if "y_m" not in D or "Ey_m" not in D:
            raise KeyError(f"{Path(mat_path).name}    'y_m'   'Ey_m' ")

        y_m  = self._as_1d(D["y_m"])
        Ey_m = self._as_1d(D["Ey_m"])

        #               None
        Ex_m =self._as_1d(D.get("Ex_m", None))
        Ez_m = self._as_1d(D.get("Ez_m", None))
        Hx_m = self._as_1d(D.get("Hx_m", None))
        Hy_m = self._as_1d(D.get("Hy_m", None))
        Hz_m = self._as_1d(D.get("Hz_m", None))

        # zeta                      Hz   Ey/Hz         None
        if "zeta_m" in D:
            zeta_m = self._as_1d(D["zeta_m"])
        elif Hz_m is not None:
            zeta_m = Ey_m / (Hz_m + 1e-30)
        else:
            zeta_m = None

        #    meta    dict 
        meta = {}
        if "meta" in D:
            try:
                md = D["meta"]
                # SciPy      dict    struct         
                if isinstance(md, np.void) and md.dtype.names:
                    for name in md.dtype.names:
                        val = md[name]
                        #     
                        while isinstance(val, np.ndarray) and val.size == 1:
                            val = val.item()
                        meta[name] = val
                elif isinstance(md, dict):
                    meta = md
            except Exception:
                pass

        return y_m, Ex_m, Ey_m, Ez_m, Hx_m, Hy_m, Hz_m, zeta_m, meta
    def eta_kspace_from_mat_and_ein(
            self,
         Ey, *,
        Hz_in=None,                    #             Hz_in    
        y_in=None,
        emode_mat_path=None,
        span_m=12e-6, origin='left',
        wavelength=None, n_medium=None, use_window=False,
        direction='+x'
    ):
        """
        1)    Emode     Hz_m     zeta_m    
        2)   /   Ein    y_i
        3)   Emode     y_i        
        4)           eta = T_forward 
           (eta, details)
        """

        Ey_in = Ey
        # 1)   /   y_i        
        y_i, Ey_in = _ensure_y_for_Ey(Ey_in, y=y_in, span_m=span_m, origin=origin)
        if Hz_in is not None:
            _, Hz_in = _ensure_y_for_Ey(np.asarray(Hz_in), y=y_i, span_m=span_m, origin=origin)

        # 2)   Emode     /   loader 
        if emode_mat_path is not None:
            emode_mat_path = Path(emode_mat_path)

            rec = self.load_emode_from_mat(emode_mat_path)
            #   loader: y, Ex,Ey,Ez,Hx,Hy,Hz,zeta,meta
            if len(rec) >= 9:
                y_m, Ex_m, Ey_m, Ez_m, Hx_m, Hy_m, Hz_m, zeta_m, meta = rec
            else:
                #   loader: y, Ey, zeta, meta, had_zeta
                y_m, Ey_m, zeta_m, meta, _ = rec
                Ex_m = Ez_m = Hx_m = Hy_m = Hz_m = None
        else:
            y_m = self.y_m
            Ey_m = self.Ey_m
            Hz_m = self.Hz_m
            zeta_m = self.zeta_m    
            meta = self.meta


        y_m = np.asarray(y_m, float)
        Ey_m = np.asarray(Ey_m, np.complex128)

        #    Hz_m    zeta_m   zeta_m   
        if Hz_m is None and zeta_m is not None:
            zeta_m = np.asarray(zeta_m, np.complex128)
            Hz_m = Ey_m / (zeta_m + 1e-30)

        # 3)   /    y_i        
        if y_m.size != y_i.size or not np.allclose([y_m[0], y_m[-1]], [y_i[0], y_i[-1]], rtol=1e-6, atol=1e-12):
            Ey_m_i =self._interp_complex(y_m, Ey_m, y_i)
            Hz_m_i = self._interp_complex(y_m, Hz_m,  y_i) if Hz_m is not None else None
            z_i    = Ey_m_i / (Hz_m_i + 1e-30) if Hz_m_i is not None else ( self._interp_complex(y_m, zeta_m, y_i) if zeta_m is not None else None )
        else:
            Ey_m_i, Hz_m_i = Ey_m, Hz_m
            z_i = (Ey_m_i / (Hz_m_i + 1e-30)) if Hz_m_i is not None else zeta_m

        if z_i is None:
            raise KeyError(f"{emode_mat_path.name}    Hz_m    zeta_m      ")

        # 4)   
        dir_sign = +1 if str(direction).strip().lower().startswith('+') else -1

        # 4)                          
        details = self.eta_kspace_vec_angler(
            y_i, Ey_in, Ey_m_i, 
            wavelength=wavelength, n_medium=n_medium
        )

        return  details
    def eta_kspace_vec_angler(
            self,
            y,                  # 1D      
            Ey_in,              #    Ey(y)
            Ey_mode,            #    Ey_mode(y)
            wavelength,         #   (m)
            n_medium,           #       n
            angler_k=None       #            W(k_y) shape (N,)
        ):

        # ----- torch backend to keep gradients -----
        dev = None
        if isinstance(Ey_in, torch.Tensor):
            dev = Ey_in.device
        elif isinstance(Ey_mode, torch.Tensor):
            dev = Ey_mode.device

        # y      
        y = _to_1d_tensor(y, dtype=torch.float64, device=dev)
        Ein = _to_complex_tensor(Ey_in, device=dev)
        Em  = _to_complex_tensor(Ey_mode, device=dev)
        dev = Ein.device
        y = y.to(device=dev, dtype=torch.float64)
        Em = Em.to(device=dev)

        # Flatten / batch handling
        if Ein.ndim == 1:
            Ein = Ein.unsqueeze(0)
        if Em.ndim == 1:
            Em = Em.unsqueeze(0)
        B = Ein.shape[0]
        if Em.shape[0] == 1 and B > 1:
            Em = Em.expand(B, -1)

        if not (y.ndim == 1 and Em.ndim == 2 and Ein.ndim == 2):
            raise ValueError("Invalid input")
        if not (y.numel() == Ein.shape[-1] == Em.shape[-1]):
            raise ValueError("Invalid input")
        N = int(y.numel())
        if N < 2:
            raise ValueError("Invalid input")

        #      
        dy = float((y[1] - y[0]).item())
        diffs = torch.diff(y)
        if not torch.allclose(diffs, torch.full_like(diffs, dy), rtol=1e-8, atol=1e-15):
            raise ValueError("Invalid input")

        # ----- k    FFT torch  -----
        # torch.fft.fftfreq       d=dy      cycles/unit      numpy   2             
        # numpy     : ky = 2*pi*fftfreq(N, d=dy)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(n=N, d=dy, dtype=torch.float64).to(dev)  # rad/m
        Ein_k = torch.fft.fft(Ein, dim=-1)   # [B,N]
        Em_k  = torch.fft.fft(Em, dim=-1)    # [B,N]

        # -----            -----
        Pink = torch.abs(Ein_k) ** 2
        Pmk  = torch.abs(Em_k)  ** 2

        # -----      Pmk                       -----
        alpha = 1.0
        eps   = 1e-300
        T = 0.1
        if angler_k is None:
            #                
            k0 = 2.0 * torch.pi / float(wavelength)
            kn = k0 * float(n_medium)
            # kz = sqrt(max(kn**2 - ky**2, 0.0))
            kz_sq = kn**2 - ky**2
            kz = torch.sqrt(torch.clamp(kz_sq, min=0.0))

            prop_only       = True      #        
            use_flux_weight = False     #          kz/kn

            maxP = Pmk.max() if Pmk.numel() > 0 else torch.tensor(0.0, dtype=Pmk.dtype, device=dev)
            W_mode = (Pmk / (maxP + eps)) ** alpha
            # threshold:      T   1.0
            W_mode = torch.where(
                W_mode >= float(T),
                torch.tensor(1.0, dtype=W_mode.dtype, device=W_mode.device),
                W_mode,
            )

            if prop_only:
                W = W_mode * (torch.abs(ky) <= kn).to(W_mode.dtype)
            else:
                W = W_mode.clone()

            if use_flux_weight:
                W = W * (kz / (kn + eps))

            W = W.to(torch.float64)
        else:
            W = _to_1d_tensor(angler_k, dtype=torch.float64, device=dev)
            if W.shape != (N,):
                raise ValueError("Invalid input")
            W = W.unsqueeze(0)

        if W.ndim == 1:
            W = W.unsqueeze(0)
        if W.shape[0] == 1 and B > 1:
            W = W.expand(B, -1)
        elif W.shape[0] not in (1, B):
            raise ValueError("Invalid W batch dimension")

        # -----           Ein_k_f = Ein_k * sqrt(W) -----
        W_clamped = torch.clamp(W, min=0.0)
        sqrtW = torch.sqrt(W_clamped)
        #   sqrtW         Ein_k         
        sqrtW_c = sqrtW.to(torch.complex128) if not torch.is_complex(sqrtW) else sqrtW
        Ein_k_f = Ein_k * sqrtW_c

        # -----      Parseval    -----
        Sw_in_f = torch.sum(torch.abs(Ein_k_f) ** 2, dim=1)  # [B]
        Sw_m    = torch.sum(torch.abs(Em_k) ** 2, dim=1)     # [B]

        overlap_k = (dy / N) * torch.sum(Ein_k_f * torch.conj(Em_k), dim=1)  # [B]

        denom = ((dy / N) * Sw_in_f) * ((dy / N) * Sw_m)
        eta_k_tensor = torch.where(
            (Sw_in_f > 0.0) & (Sw_m > 0.0),
            (torch.abs(overlap_k) ** 2).real / denom.clamp_min(1e-30),
            torch.zeros((), device=dev, dtype=torch.float64),
        )
        eta_k = eta_k_tensor.detach().cpu().float()

        # projection power (keeps amplitude info; not normalized by Ein power)
        P_mode = (dy / N) * Sw_m                               # (B,)
        proj_power_tensor = (torch.abs(overlap_k) ** 2).real / P_mode.clamp_min(1e-30)  # (B,)
        proj_power = proj_power_tensor.detach().cpu().float()

        # -----         -----
        sum_prod_raw = torch.sum(W * Pink * Pmk, dim=1).detach().cpu().float()
        power_parseval = (((dy / N) * torch.sum(torch.abs(Ein_k_f) ** 2, dim=1))).detach().cpu().float()
        sim_norm = eta_k

        return {
            "ky": ky,                                # torch.tensor
            "Pink": Pink,                            # torch.tensor
            "Pmk": Pmk,                              # torch.tensor
            "W": W,                                  # torch.tensor
            "overlap_k": overlap_k,                  # complex torch.tensor (  )
            "eta_k": eta_k,                          # torch float tensor (B,)
            "eta_k_tensor": eta_k_tensor,            # torch tensor (B,) keeps grad
            "proj_power": proj_power,                # torch float tensor (B,) detached
            "proj_power_tensor": proj_power_tensor,  # torch tensor (B,) keeps grad
            "sim_norm": sim_norm,                    # torch float tensor (B,)
            "power_parseval": power_parseval,        # torch float tensor (B,)
            "sum_prod_raw": sum_prod_raw,            # torch float tensor (B,)
            "dy": dy,                                # python float
            "N": N,                                  # python int
            "wavelength": float(wavelength),
            "n_medium": float(n_medium),
        }
    def eta_kspace_vec_angler_(
        self,
        y,                  # 1D      
        Ey_in,              #    Ey(y)
        Ey_mode,            #    Ey_mode(y)
        wavelength,         #   (m)
        n_medium,           #       n
        angler_k=None       #            W(k_y) shape (N,)
    ):

        #       
        y   = np.asarray(y, float).squeeze()
        Ein = np.asarray(Ey_in,  np.complex128).squeeze()
        Em  = np.asarray(Ey_mode,np.complex128).squeeze()
        if not (y.ndim == Ein.ndim == Em.ndim == 1):
            raise ValueError("Invalid input")
        if not (y.size == Ein.size == Em.size):
            raise ValueError("Invalid input")
        N = y.size
        if N < 2:
            raise ValueError("Invalid input")
        dy = float(y[1]-y[0])
        if not np.allclose(np.diff(y), dy, rtol=1e-8, atol=1e-15):
            raise ValueError("Invalid input")

        #   k    FFT numpy      
        ky    = 2*np.pi*np.fft.fftfreq(N, d=dy)    # rad/m
        Ein_k = np.fft.fft(Ein)                    #    dy
        Em_k  = np.fft.fft(Em)

        #               
        Pink = np.abs(Ein_k)**2
        Pmk  = np.abs(Em_k )**2

        #        Pmk                        
        #     W_mode = (Pmk/max(Pmk))**alpha    alpha        
        alpha =1
        eps   = 1e-300
        T=0.1
        if angler_k is None:
            #                
            k0 = 2*np.pi/float(wavelength)
            kn = k0*float(n_medium)
            kz = np.sqrt(np.clip(kn**2 - ky**2, 0.0, None))

            prop_only      = True      #        
            use_flux_weight= False     #          kz/kn

            W_mode = (Pmk / (Pmk.max() + eps))**alpha
            W_mode=np.where(W_mode >= T, 1.0, W_mode) 
            if prop_only:
                W = W_mode * (np.abs(ky) <= kn)
            else:
                W = W_mode.copy()
            if use_flux_weight:
                W = W * (kz/(kn + eps))
            W = np.asarray(W, float)
        else:
            W = np.asarray(angler_k, float).squeeze()
            if W.shape != (N,):
                raise ValueError("Invalid input")

        #             Ein_k_f = Ein_k * sqrt(W)  
        Ein_k_f = Ein_k * np.sqrt(np.maximum(W, 0.0))

        #        Parseval     
        Sw_in_f= float(np.sum(np.abs(Ein_k_f)**2))
        Sw_m    = float(np.sum(np.abs(Em_k   )**2))

        overlap_k = (dy / N) * np.sum(Ein_k_f * np.conj(Em_k))  #      Ein_f * Em* dy

        if Sw_in_f > 0.0 and Sw_m > 0.0:
            eta_k = float(np.abs(overlap_k)**2 / ((dy/N)*Sw_in_f * (dy/N)*Sw_m))
        else:
            eta_k = 0.0

        #            
        sum_prod_raw   = float(np.sum(W * Pink * Pmk))                 #    
        power_parseval = float((dy / N) * np.sum(np.abs(Ein_k_f)**2))  #        
        sim_norm       = eta_k                                         #     

        return                                                       {
            "ky": ky,
            "Pink": Pink,
            "Pmk": Pmk,
            "W": W,
            "overlap_k": overlap_k,
            "eta_k": eta_k,
            "sim_norm": sim_norm,
            "power_parseval": power_parseval,
            "sum_prod_raw": sum_prod_raw,  #          
            "dy": dy,
            "N": N,
            "wavelength": float(wavelength),
            "n_medium": float(n_medium),
        }
    def eta_real_from_mat_and_ein(self,
        Ex, Ey, *,
        Hz_in=None,                    #             Hz_in    
        y_in=None,
        emode_mat_path=None,
        span_m=12e-6, origin='left',
        wavelength=None, n_medium=None, use_window=False,
        direction='+x'
    ):
        """
        1)    Emode     Hz_m     zeta_m    
        2)   /   Ein    y_i
        3)   Emode     y_i        
        4)           eta = T_forward 
           (eta, details)
        """
        # 0)    Ein    torch/np        Ey Ex    
        try:
            import torch
            Ex = Ex.detach().cpu().numpy() if torch.is_tensor(Ex) else Ex
            Ey = Ey.detach().cpu().numpy() if torch.is_tensor(Ey) else Ey
            if Hz_in is not None and torch.is_tensor(Hz_in):
                Hz_in = Hz_in.detach().cpu().numpy()
        except Exception:
            pass
        Ex_in = Ex
        Ey_in =Ey

        # 1)   /   y_i        
        y_i, Ex_in = _ensure_y_for_Ey(Ex_in, y=y_in, span_m=span_m, origin=origin)
        y_i, Ey_in = _ensure_y_for_Ey(Ey_in, y=y_in, span_m=span_m, origin=origin)
        if Hz_in is not None:
            _, Hz_in = _ensure_y_for_Ey(np.asarray(Hz_in), y=y_i, span_m=span_m, origin=origin)

        if emode_mat_path is not None:
            emode_mat_path = Path(emode_mat_path)

            rec = self.load_emode_from_mat(emode_mat_path)
            #   loader: y, Ex,Ey,Ez,Hx,Hy,Hz,zeta,meta
            if len(rec) >= 9:
                y_m, Ex_m, Ey_m, Ez_m, Hx_m, Hy_m, Hz_m, zeta_m, meta = rec
            else:
                #   loader: y, Ey, zeta, meta, had_zeta
                y_m, Ey_m, zeta_m, meta, _ = rec
                Ex_m = Ez_m = Hx_m = Hy_m = Hz_m = None
        else:
            y_m = self.y_m
            Ey_m = self.Ey_m
            Hz_m = self.Hz_m
            zeta_m = self.zeta_m    
            meta = self.meta

        y_m = np.asarray(y_m, float)
        Ey_m = np.asarray(Ey_m, np.complex128)

        #    Hz_m    zeta_m   zeta_m   
        if Hz_m is None and zeta_m is not None:
            zeta_m = np.asarray(zeta_m, np.complex128)
            Hz_m = Ey_m / (zeta_m + 1e-30)

        # 3)   /    y_i        
        if y_m.size != y_i.size or not np.allclose([y_m[0], y_m[-1]], [y_i[0], y_i[-1]], rtol=1e-6, atol=1e-12):
            Ey_m_i = self._interp_complex(y_m, Ey_m, y_i)
            Hz_m_i = self._interp_complex(y_m, Hz_m,  y_i) if Hz_m is not None else None
            z_i    = Ey_m_i / (Hz_m_i + 1e-30) if Hz_m_i is not None else ( self._interp_complex(y_m, zeta_m, y_i) if zeta_m is not None else None )
        else:
            Ey_m_i, Hz_m_i = Ey_m, Hz_m
            z_i = (Ey_m_i / (Hz_m_i + 1e-30)) if Hz_m_i is not None else zeta_m

        if z_i is None:
            raise KeyError(f"{emode_mat_path.name}    Hz_m    zeta_m      ")

        # 4)   
        dir_sign = +1 if str(direction).strip().lower().startswith('+') else -1

        # 5)         eta   T_forward 
        eta, details = self.eta_real_vec_weighted(
            y_i, Ey_in, Ex_in,
            Ey_m_i, z_i,
            wavelength=wavelength, n_medium=n_medium,
            use_window=use_window,
            Hz_mode_in=Hz_m_i,
            Hz_in_in=Hz_in,
            dir_in=dir_sign, dir_mode=dir_sign,
            auto_align=True
        )
        details = dict(details, emode_mat=str(emode_mat_path), span_m=float(y_i[-1] - y_i[0]))
        return eta, details
    def _trapz_cplx(self,f, y):
        #         /      trapz
        return np.trapz(f, y)
    def eta_real_vec_weighted(self,
        y,
        Ey_in, Ex_in,                 # Ex_in                 
        Ey_mode, zeta_m,
        wavelength, n_medium,
        use_window=False,
        Hz_mode_in=None,              #          Hz
        Hz_in_in=None,                #             Hz
        dir_in=+1, dir_mode=+1,
        auto_align=True,
        beta_in=None,
        y_dir=+1,
        ex_shift=0.0,
    ):
        """
                Lumerical   mode expansion/overlap      
        -    TE     x        ~ Ey * Hz*
        -    eta = T_forward         details    I1,I2,N,P,a,b  
        """
        #   
        c0  = 299792458.0
        mu0 = 4*np.pi*1e-7
        Z0  = mu0 * c0
        k0  = 2*np.pi/float(wavelength)

        #     
        y       = self._as_1d(y, float)
        Ey_in   = self._as_1d(Ey_in, np.complex128)
        Ey_mode = self._as_1d(Ey_mode, np.complex128)
        zeta_m  = self._as_1d(zeta_m, np.complex128)
        if not (y.size == Ey_in.size == Ey_mode.size == zeta_m.size):
            raise ValueError("Invalid input")
        Np = y.size
        if Np < 2:
            raise ValueError("Invalid input")

        #    
        w = np.hanning(Np) if use_window else np.ones(Np)

        #    Hz        Hz_mode_in     zeta_m   
        if Hz_mode_in is None:
            Hz_mode = -(dir_mode) * Ey_mode / (zeta_m + 1e-30)
        else:
            Hz_mode = (dir_mode) * self._as_1d(Hz_mode_in, np.complex128)
            if Hz_mode.size != Np:
                raise ValueError("Invalid input")

        #    Hz       /     Hz_in_in        
        if Hz_in_in is None:
            if n_medium is None:
                raise ValueError("Invalid input")
            eta_med = Z0 / float(n_medium)
            Hz_in = (dir_in) * Ey_in / eta_med
        else:
            Hz_in = (dir_in) * self._as_1d(Hz_in_in, np.complex128)
            if Hz_in.size != Np:
                raise ValueError("Invalid input")

        #   
        Ey_in_w   = Ey_in * w
        Ey_mode_w = Ey_mode * w
        Hz_in_w   = Hz_in * w
        Hz_mode_w = Hz_mode * w

        #      1/2  Pin, Nm   +x    
        #                     TE    Ey*Hz* 
        P_nohalf  = self._trapz_cplx(Ey_in_w  * np.conj(Hz_in_w),  y)      #   E_in   H_in*
        N_nohalf  = self._trapz_cplx(Ey_mode_w* np.conj(Hz_mode_w), y)      #   E_m    H_m*
        Pin = 0.5 * P_nohalf
        Nm  = 0.5 * N_nohalf

        #           Re(Pin), Re(Nm) > 0 
        flipped_in = flipped_md = False
        if auto_align and np.real(Pin) < 0:
            Hz_in_w  = -Hz_in_w
            Pin      = -Pin
            P_nohalf = -P_nohalf
            flipped_in = True
        if auto_align and np.real(Nm) < 0:
            Hz_mode_w = -Hz_mode_w
            Nm        = -Nm
            N_nohalf  = -N_nohalf
            flipped_md = True

        #              
        I1 = self._trapz_cplx(Ey_in_w   * np.conj(Hz_mode_w), y)   #   E_in   H_m*
        I2 = self._trapz_cplx(np.conj(Ey_mode_w) * Hz_in_w,   y)   #   E_m*   H_in

        #      a,b   1/4 
        a = 0.25 * ( I1/Nm + I2/np.conj(Nm) )
        b = 0.25 * ( I1/Nm - I2/np.conj(Nm) )

        #        T_forward    eta    
        T_forward = (np.abs(a)**2) * (np.real(Nm) / (np.real(Pin) + 1e-30))
        eta = float(np.real(T_forward))

        #     overlap                    
        # overlap   | Re( (I1 * I2_conj) / (N_nohalf * Re(P_nohalf)) ) |
        eta_overlap = float(
            np.abs(
                np.real( (I1 * np.conj(I2)) / ((N_nohalf + 1e-30) * (np.real(P_nohalf) + 1e-30)) )
            )
        )

        #     
        dy_arr = np.diff(y)
        details = dict(
            used_window=bool(use_window),
            dir_in_used=int(dir_in), dir_mode_used=int(dir_mode),
            flipped_in=bool(flipped_in), flipped_mode=bool(flipped_md),
            Pin=float(np.real(Pin)), Pm=float(np.real(Nm)),   #    Pm=Re(Nm)     
            I1=complex(I1), I2=complex(I2), Nm=complex(Nm), P_in=complex(Pin),
            a=complex(a), b=complex(b),
            T_forward=float(np.real(T_forward)),
            eta_overlap=eta_overlap,
            y_uniform=bool(np.allclose(dy_arr, dy_arr[0], rtol=1e-6, atol=1e-12)) if dy_arr.size else True,
            dy_mean=float(np.mean(dy_arr)) if dy_arr.size else np.nan,
            k0=k0, Z0=Z0, beta_used=(float(beta_in) if beta_in is not None else None),
        )
        return eta, details



    












