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
    n1: Optional[int] = 0
    n2: Optional[int] = 0
    n3: Optional[int] = 0
    nVal: Optional[int] = 0
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
    nValue: int
    n60: Optional[int] = None
    tcrPct: float
    scrPct: float
    rqdPct: float
    effectiveOverburdenStress: float
    Cn: float
    N1_E: float
    N1_final_dilatancy: float

@app.post("/api/calculate", response_model=List[CalculatedRecordResponse])
def calculate_borelog(request: BorelogCalculationRequest):
    """
    Computes all geotechnical parameters:
    - Sample Recovery (%) = (Recovered / Interval) * 100
    - N = N2 + N3
    - TCR (%), SCR (%) [with SCR <= TCR validation], RQD (%)
    - N60 = (Er / 60) * N  [Er restricted to <= 90%]
    - CN = 0.77 * log10(20 * Pa / sigma_v')  [0.40 <= CN <= 2.00, Pa = 101.30269 kPa]
    - (N1)E = CN * N60
    - Dilatancy Correction N1'' = 15 + 0.5 * ((N1)E - 15) for (N1)E > 15
    """
    # Enforce Drive Energy Limit <= 90%
    Er = min(request.energyEfficiency, 90.0)
    Pa = 101.30269  # Atmospheric Pressure in kPa
    gwt = request.waterTableDepth

    results = []
    
    for r in request.records:
        interval = max(0.01, r.depthTo - r.depthFrom)
        
        # 1. Sample Recovery
        sample_rec_pct = min(100.0, (r.recoveredLength / interval) * 100.0)
        
        # 2. N Value Calculation
        n_val = r.n2 + r.n3 if r.sampleType == 'SPT' else r.nVal
        
        # 3. Rock Core Recoveries
        tcr_pct = 0.0
        scr_pct = 0.0
        rqd_pct = 0.0
        if r.sampleType == 'CR':
            tcr_pct = min(100.0, (r.coreRec / interval) * 100.0)
            # Validation: SCR <= TCR
            raw_scr = (r.solidCore / interval) * 100.0
            scr_pct = min(tcr_pct, raw_scr)
            
            # RQD = (Sum of pieces >= 0.10m / interval) * 100
            sum_pieces = 0.0
            if r.rockPieces:
                for piece in r.rockPieces.split(','):
                    try:
                        val = float(piece.strip())
                        if val >= 0.10:  # Sound rock piece >= 10 cm
                            sum_pieces += val
                    except ValueError:
                        pass
            rqd_pct = min(100.0, (sum_pieces / interval) * 100.0)
        
        # 4. Energy Efficiency Correction (N60)
        n60 = round((Er / 60.0) * n_val) if r.sampleType == 'SPT' else None
        
        # 5. Overburden Correction (CN & (N1)E)
        mid_depth = (r.depthFrom + r.depthTo) / 2.0
        
        # Effective unit weight adjustment for Water Table
        if mid_depth > gwt:
            effective_gamma = max(0.1, r.unitWeight - 9.81)
        else:
            effective_gamma = r.unitWeight
            
        sigma_v_prime = max(1.0, effective_gamma * mid_depth)
        
        Cn = 0.77 * math.log10((20.0 * Pa) / sigma_v_prime)
        # Restriction: 0.40 <= CN <= 2.00
        Cn = max(0.40, min(2.00, Cn))
        
        N1_E = Cn * (n60 if n60 is not None else n_val)
        
        # 6. Dilatancy Correction (N1'')
        if N1_E <= 15.0:
            N1_final = N1_E
        else:
            N1_final = 15.0 + 0.5 * (N1_E - 15.0)

        results.append(CalculatedRecordResponse(
            id=r.id,
            sampleDepthInterval=round(interval, 2),
            sampleRecoveryPct=round(sample_rec_pct, 1),
            nValue=n_val,
            n60=n60,
            tcrPct=round(tcr_pct, 1),
            scrPct=round(scr_pct, 1),
            rqdPct=round(rqd_pct, 1),
            effectiveOverburdenStress=round(sigma_v_prime, 2),
            Cn=round(Cn, 3),
            N1_E=round(N1_E, 1),
            N1_final_dilatancy=round(N1_final, 1)
        ))

    return results

@app.post("/api/spt-graph")
def generate_spt_graph(request: BorelogCalculationRequest):
    """
    Generates a high-resolution Matplotlib SPT Depth Profile Graph matching
    the exact visual styling of reference image:
    - Top X-axis labeled "SPT 'N' Value" (0 to 120)
    - Inverted Y-axis labeled "Depth below Seabed, m" (0 to 30m)
    - Blue curve with solid circles for Field N
    - Orange curve with solid circles for Corrected N1''
    - Light dashed grid
    - Inside Legend top right
    - Inside Bold Title "Borehole: BH-XX" at bottom left
    """
    calcs = calculate_borelog(request)
    
    depths = []
    n_field = []
    n_corrected = []

    for idx, r in enumerate(request.records):
        if r.sampleType == 'SPT':
            mid_d = (r.depthFrom + r.depthTo) / 2.0
            depths.append(mid_d)
            n_field.append(r.n2 + r.n3)
            n_corrected.append(calcs[idx].N1_final_dilatancy)

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

            ax.plot(n_field_smooth, depths_smooth, color='#4A90E2', linestyle='-', linewidth=2.0, label='SPT (N field)')
            ax.plot(n_corr_smooth, depths_smooth, color='#E67E22', linestyle='-', linewidth=2.0, label='SPT (N Corrected)')
        except Exception:
            ax.plot(n_field, depths, color='#4A90E2', linestyle='-', label='SPT (N field)')
            ax.plot(n_corrected, depths, color='#E67E22', linestyle='-', label='SPT (N Corrected)')

    ax.scatter(n_field, depths, color='#4A90E2', s=35, zorder=5)
    ax.scatter(n_corrected, depths, color='#E67E22', s=35, zorder=5)

    # Invert Y-axis & set bounds
    ax.set_ylim(30, 0)
    ax.set_xlim(0, 120)

    # Move X-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_xlabel("SPT 'N' Value", fontsize=11, fontweight='bold', family='serif', labelpad=10)
    ax.set_ylabel("Depth below Seabed, m", fontsize=10, family='serif', labelpad=8)
    
    ax.grid(True, linestyle='--', color='#e2e8f0', alpha=0.8)
    ax.legend(loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='#cbd5e1')

    # Add Borehole ID at bottom left inside plot
    bh_no = request.boreholeNo or 'BH-01'
    ax.text(0.08, 0.08, f"Borehole: {bh_no}", transform=ax.transAxes, fontsize=14, fontweight='bold', family='serif')

    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=180)
    plt.close(fig)
    img_buf.seek(0)

    return StreamingResponse(img_buf, media_type="image/png")