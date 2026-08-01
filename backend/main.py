import io
import math
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

app = FastAPI(
    title="Geotechnical Borelog Calculation & Graph Engine API",
    description="Calculates SPT corrections (N60, Overburden, Dilatancy), Rock Core Recoveries (TCR, SCR, RQD), and generates SPT Depth Profile Plots.",
    version="1.0.0"
)

# Enable CORS for Vercel Frontend Connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SampleRecord(BaseModel):
    id: str
    depthFrom: float = Field(..., description="Depth From in meters")
    depthTo: float = Field(..., description="Depth To in meters")
    cdFrom: Optional[str] = ""
    cdTo: Optional[str] = ""
    sampleType: str = Field(..., description="DS, SPT, UDS, or CR")
    sampleNo: int = Field(..., description="Numeric Sample Number")
    recoveredLength: float = Field(0.0, description="Recovered sample in meters")
    n1: Optional[str] = "0"
    n2: Optional[str] = "0"
    n3: Optional[str] = "0"
    nVal: Optional[str] = "0"
    hatchType: Optional[str] = "CLAY"
    unitWeight: float = Field(18.0, description="Unit weight of soil gamma in kN/m3")
    description: Optional[str] = ""
    coreRec: Optional[float] = 0.0
    solidCore: Optional[float] = 0.0
    rockPieces: Optional[str] = ""

class BorelogCalculationRequest(BaseModel):
    mode: str = Field("field", description="'field' or 'final'")
    waterTableDepth: float = Field(2.5, description="Ground water table in meters")
    energyEfficiency: float = Field(60.0, description="Drive energy efficiency Er (%) restricted to 90%")
    boreholeNo: Optional[str] = "BH-01"
    records: List[SampleRecord]

class CalculatedRecordResponse(BaseModel):
    id: str
    sampleDepthInterval: float
    sampleRecoveryPct: float
    nValue: str
    n60: Optional[str] = None
    tcrPct: float
    scrPct: float
    rqdPct: float
    effectiveOverburdenStress: float
    Cn: Optional[float] = None
    N1_E: Optional[str] = None
    N1_final_dilatancy: Optional[str] = None

@app.post("/api/calculate", response_model=List[CalculatedRecordResponse])
def calculate_borelog(request: BorelogCalculationRequest):
    """
    Computes all geotechnical parameters with cumulative overburden stress:
    - Cumulative effective overburden stress sigma_v' summed down through all preceding strata layers.
    - SPT Corrections (N60, Cn, N1_E, Dilatancy N1'') applied ONLY to SPT samples.
    - Handles 'R' / 'Refusal' by setting N'' as 'Refusal' and using 100 as the final output value for graphs/computations.
    - Non-SPT samples (DS, UDS, CR) contribute to overburden stress accumulation but ignore SPT corrections.
    """
    Er = min(request.energyEfficiency, 90.0)
    Pa = 101.30269  # Atmospheric Pressure in kPa
    gwt = request.waterTableDepth

    # Sort records by depthFrom to ensure sequential stress accumulation from top to bottom
    sorted_records = sorted(request.records, key=lambda x: x.depthFrom)

    current_depth = 0.0
    current_sigma_v_top = 0.0

    calc_map = {}

    for r in sorted_records:
        interval = max(0.01, r.depthTo - r.depthFrom)
        
        # 1. Fill gap if depthFrom > current_depth
        if r.depthFrom > current_depth:
            gap = r.depthFrom - current_depth
            gap_mid = current_depth + gap / 2.0
            gap_gamma = (r.unitWeight - 9.81) if gap_mid > gwt else r.unitWeight
            current_sigma_v_top += max(0.1, gap_gamma) * gap
            current_depth = r.depthFrom

        # 2. Cumulative Effective Overburden Stress at Midpoint & Bottom of current layer
        half_interval = interval / 2.0
        mid_depth = r.depthFrom + half_interval

        if mid_depth > gwt:
            eff_gamma = max(0.1, r.unitWeight - 9.81)
        else:
            eff_gamma = r.unitWeight

        sigma_v_prime_mid = current_sigma_v_top + eff_gamma * half_interval
        sigma_v_prime_bottom = current_sigma_v_top + eff_gamma * interval

        # Advance top stress for next layer
        current_sigma_v_top = sigma_v_prime_bottom
        current_depth = r.depthTo

        # Ensure positive stress for log calculation
        sigma_v_prime = max(1.0, sigma_v_prime_mid)

        # 3. Sample Recovery
        sample_rec_pct = min(100.0, (r.recoveredLength / interval) * 100.0)

        # 4. Rock Core Recoveries
        tcr_pct = 0.0
        scr_pct = 0.0
        rqd_pct = 0.0
        if r.sampleType == 'CR':
            tcr_pct = min(100.0, (r.coreRec / interval) * 100.0)
            raw_scr = (r.solidCore / interval) * 100.0
            scr_pct = min(tcr_pct, raw_scr)

            sum_pieces = 0.0
            if r.rockPieces:
                for piece in r.rockPieces.split(','):
                    try:
                        val = float(piece.strip())
                        if val >= 0.10:
                            sum_pieces += val
                    except ValueError:
                        pass
            rqd_pct = min(100.0, (sum_pieces / interval) * 100.0)

        # 5. SPT Corrections (ONLY for SPT samples)
        n_val_str = "0"
        n60_str = None
        Cn = None
        N1_E_str = None
        N1_final_str = None

        if r.sampleType == 'SPT':
            n2_str = str(r.n2).strip()
            n3_str = str(r.n3).strip()
            
            is_refusal = (n2_str.upper() in ['R', 'REFUSAL'] or n3_str.upper() in ['R', 'REFUSAL'] or str(r.nVal).strip().upper() in ['R', 'REFUSAL'])

            if is_refusal:
                n_val_str = "R"
                n60_str = "R"
                N1_E_str = "Refusal"
                N1_final_str = "Refusal"
            else:
                try:
                    val2 = int(n2_str) if n2_str else 0
                    val3 = int(n3_str) if n3_str else 0
                    n_num = val2 + val3
                except ValueError:
                    n_num = int(r.nVal) if r.nVal and str(r.nVal).isdigit() else 0

                n_val_str = str(n_num)
                n60_val = round((Er / 60.0) * n_num)
                n60_str = str(n60_val)

                Cn_raw = 0.77 * math.log10((20.0 * Pa) / sigma_v_prime)
                Cn = max(0.40, min(2.00, Cn_raw))

                N1_E_raw = Cn * n60_val
                N1_E_str = str(round(N1_E_raw, 1))

                if N1_E_raw <= 15.0:
                    N1_final_raw = N1_E_raw
                else:
                    N1_final_raw = 15.0 + 0.5 * (N1_E_raw - 15.0)

                N1_final_str = str(round(N1_final_raw, 1))

        calc_map[r.id] = CalculatedRecordResponse(
            id=r.id,
            sampleDepthInterval=round(interval, 2),
            sampleRecoveryPct=round(sample_rec_pct, 1),
            nValue=n_val_str,
            n60=n60_str,
            tcrPct=round(tcr_pct, 1),
            scrPct=round(scr_pct, 1),
            rqdPct=round(rqd_pct, 1),
            effectiveOverburdenStress=round(sigma_v_prime, 2),
            Cn=round(Cn, 3) if Cn is not None else None,
            N1_E=N1_E_str,
            N1_final_dilatancy=N1_final_str
        )

    # Return results in the original request order
    return [calc_map[r.id] for r in request.records]

@app.post("/api/spt-graph")
def generate_spt_graph(request: BorelogCalculationRequest):
    """
    Generates a high-resolution Matplotlib SPT Depth Profile Graph matching
    the exact visual styling of reference image:
    - Top X-axis labeled 'SPT 'N' & Corrected 'N₁' Value' (0 to 100)
    - Inverted Y-axis labeled 'Depth below Ground Level (m)' (0 to 25m)
    - Handles 'Refusal' (R) by mapping value to 100 for graph plotting.
    """
    calcs = calculate_borelog(request)
    
    depths = []
    n_field = []
    n_corrected = []

    for idx, r in enumerate(request.records):
        if r.sampleType == 'SPT':
            mid_d = (r.depthFrom + r.depthTo) / 2.0
            depths.append(mid_d)
            
            # Field N value parsing with Refusal check
            n2_s = str(r.n2).strip().upper()
            n3_s = str(r.n3).strip().upper()
            if n2_s in ['R', 'REFUSAL'] or n3_s in ['R', 'REFUSAL'] or str(r.nVal).strip().upper() in ['R', 'REFUSAL']:
                n_field.append(100.0)
            else:
                try:
                    n_field.append(float(r.n2) + float(r.n3))
                except ValueError:
                    n_field.append(float(r.nVal) if str(r.nVal).replace('.','',1).isdigit() else 50.0)

            # Corrected N value parsing with Refusal check
            corr_str = str(calcs[idx].N1_final_dilatancy).strip().upper()
            if corr_str in ['REFUSAL', 'R']:
                n_corrected.append(100.0)
            else:
                try:
                    n_corrected.append(float(corr_str))
                except ValueError:
                    n_corrected.append(0.0)

    fig, ax = plt.subplots(figsize=(6, 7), dpi=180)
    
    if len(depths) >= 2:
        depths_arr = np.array(depths)
        n_field_arr = np.array(n_field)
        n_corr_arr = np.array(n_corrected)

        depths_smooth = np.linspace(depths_arr.min(), depths_arr.max(), 200)
        
        try:
            spl_field = make_interp_spline(depths_arr, n_field_arr, k=min(3, len(depths)-1))
            spl_corr = make_interp_spline(depths_arr, n_corr_arr, k=min(3, len(depths)-1))
            
            n_field_smooth = spl_field(depths_smooth)
            n_corr_smooth = spl_corr(depths_smooth)

            ax.plot(n_field_smooth, depths_smooth, color='#2563eb', linestyle='-', linewidth=2.0, label='SPT N Value')
            ax.plot(n_corr_smooth, depths_smooth, color='#d97706', linestyle='-', linewidth=2.0, label='Corrected N₁ Value')
        except Exception:
            ax.plot(n_field, depths, color='#2563eb', linestyle='-', label='SPT N Value')
            ax.plot(n_corrected, depths, color='#d97706', linestyle='-', label='Corrected N₁ Value')

    ax.scatter(n_field, depths, color='#2563eb', s=35, zorder=5)
    ax.scatter(n_corrected, depths, color='#d97706', s=35, zorder=5)

    # Invert Y-axis & set bounds
    ax.set_ylim(25, 0)
    ax.set_xlim(0, 100)

    # Move X-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_xlabel("SPT 'N' & Corrected 'N₁' Value", fontsize=11, fontweight='bold', family='serif', labelpad=10)
    ax.set_ylabel("Depth below Ground Level (m)", fontsize=10, family='serif', labelpad=8)
    
    ax.grid(True, linestyle='--', color='#e2e8f0', alpha=0.8)
    ax.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='#cbd5e1')

    # Add Borehole ID at bottom left inside plot
    bh_no = request.boreholeNo or 'BH-01'
    ax.text(0.08, 0.08, f"Borehole: {bh_no} (Continuous N & N₁ Profiles)", transform=ax.transAxes, fontsize=11, fontweight='bold', family='serif')

    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=180)
    plt.close(fig)
    img_buf.seek(0)

    return StreamingResponse(img_buf, media_type="image/png")
