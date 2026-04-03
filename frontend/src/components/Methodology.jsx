import React, { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

function Latex({ math, display = false }) {
  const ref = useRef()
  useEffect(() => {
    if (ref.current) {
      katex.render(math, ref.current, { displayMode: display, throwOnError: false })
    }
  }, [math, display])
  return <span ref={ref} />
}

function Block({ math }) {
  return (
    <div style={{ margin: '1rem 0', padding: '1rem', background: 'var(--bg)', borderRadius: 6, overflowX: 'auto' }}>
      <Latex math={math} display />
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="card">
      <div className="card-header"><h2>{title}</h2></div>
      <div style={{ color: 'var(--text)', lineHeight: 1.9, fontSize: '0.95rem' }}>{children}</div>
    </div>
  )
}

export default function Methodology() {
  return (
    <>
      <div className="page-header">
        <h1>Methodology</h1>
        <p>Mathematical foundations of the ELO and Glicko-2 rating systems for snooker</p>
      </div>

      <Section title="1. ELO Rating System (Adapted for Snooker)">
        <p>
          Unlike chess ELO which models match outcomes, our system models <strong>frame win probability</strong>.
          This is the key adaptation: snooker matches consist of multiple frames, and the frame-level model
          enables predicting exact match scores across any format (Best-of-5 to Best-of-35).
        </p>

        <h3 style={{ marginTop: '1.5rem', color: '#fff' }}>Expected Frame Win Rate</h3>
        <p>
          Given players with ratings <Latex math="R_1" /> and <Latex math="R_2" />,
          the expected probability that Player 1 wins a frame is:
        </p>
        <Block math="E_1 = \frac{1}{1 + e^{(R_2 - R_1) / d}}" />
        <p>
          where <Latex math="d = 327.15" /> is the scaling divisor (MLE-optimized; chess uses 400).
        </p>

        <h3 style={{ marginTop: '1.5rem', color: '#fff' }}>Rating Update</h3>
        <p>After a match with frame scores <Latex math="s_1" /> and <Latex math="s_2" />:</p>
        <Block math="R_1^{\text{new}} = R_1 + K \cdot (s_1 + s_2) \cdot (S_1 - E_1)" />
        <p>where:</p>
        <ul style={{ paddingLeft: '1.5rem', margin: '0.5rem 0' }}>
          <li><Latex math="K = 9.77" /> is the adjustment factor (MLE-optimized; typically 8-20)</li>
          <li><Latex math="S_1 = \frac{s_1}{s_1 + s_2}" /> is the actual frame win rate</li>
          <li><Latex math="E_1" /> is the expected frame win rate</li>
          <li>The update is weighted by total frames <Latex math="(s_1 + s_2)" />, so longer matches have more impact</li>
        </ul>

        <h3 style={{ marginTop: '1.5rem', color: '#fff' }}>Match Win Probability (Binomial Model)</h3>
        <p>
          Given frame win probability <Latex math="p" /> and a Best-of-<Latex math="N" /> format
          (first to <Latex math="w = \lceil N/2 \rceil" />), the match win probability is:
        </p>
        <Block math="P(\text{win match}) = \sum_{k=w}^{N} \binom{k-1}{w-1} \cdot p^{w-1} \cdot (1-p)^{k-w} \cdot p" />
        <p>
          This sums over all possible winning scorelines. For example, in Best-of-9 (first to 5),
          it computes <Latex math="P(5\text{-}0) + P(5\text{-}1) + P(5\text{-}2) + P(5\text{-}3) + P(5\text{-}4)" />.
        </p>
        <p style={{ marginTop: '0.75rem' }}>
          A 56% frame edge becomes a 68% match win probability in BO17, and 75% in BO35.
          Longer formats amplify skill differences — this is why the World Championship uses BO35 finals.
        </p>
      </Section>

      <Section title="2. Glicko-2 Rating System">
        <p>
          Glicko-2 (Glickman, 2001) extends ELO with two additional parameters per player:
        </p>

        <div className="grid-2" style={{ margin: '1rem 0' }}>
          <div className="stat" style={{ textAlign: 'left', padding: '1rem' }}>
            <div style={{ color: 'var(--blue)', fontWeight: 700 }}>Rating Deviation (RD / <Latex math="\phi" />)</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
              Uncertainty in the rating. High RD = less confident (new/inactive players).
              Decreases with more games, increases with inactivity.
            </div>
          </div>
          <div className="stat" style={{ textAlign: 'left', padding: '1rem' }}>
            <div style={{ color: 'var(--blue)', fontWeight: 700 }}>Volatility (<Latex math="\sigma" />)</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
              How consistent a player's performance is. High volatility = unpredictable results.
            </div>
          </div>
        </div>

        <h3 style={{ marginTop: '1.5rem', color: '#fff' }}>Update Algorithm</h3>
        <p>Convert to Glicko-2 scale:</p>
        <Block math="\mu = \frac{R - 1500}{173.72}, \quad \phi = \frac{\text{RD}}{173.72}" />

        <p>Compute variance <Latex math="v" /> and performance delta <Latex math="\Delta" /> from opponents:</p>
        <Block math="v = \left[ \sum_{j} g(\phi_j)^2 \cdot E_j \cdot (1 - E_j) \right]^{-1}" />
        <Block math="\Delta = v \sum_{j} g(\phi_j)(s_j - E_j)" />

        <p>
          where <Latex math="g(\phi) = \frac{1}{\sqrt{1 + 3\phi^2/\pi^2}}" /> reduces the impact
          of uncertain opponents, and <Latex math="s_j" /> is the frame win proportion against opponent <Latex math="j" />.
        </p>

        <p>Update volatility via the Illinois algorithm, then:</p>
        <Block math="\phi^* = \sqrt{\phi^2 + \sigma_{\text{new}}^2}" />
        <Block math="\phi_{\text{new}} = \frac{1}{\sqrt{1/{\phi^*}^2 + 1/v}}, \quad \mu_{\text{new}} = \mu + \phi_{\text{new}}^2 \sum_j g(\phi_j)(s_j - E_j)" />

        <p>
          The system volatility constraint <Latex math="\tau = 1.488" /> (MLE-optimized) controls
          how much <Latex math="\sigma" /> can change between rating periods.
        </p>
      </Section>

      <Section title="3. Parameter Optimization (MLE)">
        <p>
          Parameters are chosen by <strong>Maximum Likelihood Estimation</strong>, not manual tuning.
          We maximize the log-likelihood of all observed frame outcomes:
        </p>
        <Block math="\mathcal{L} = \sum_{i} \left[ s_{1,i} \cdot \ln(p_i) + s_{2,i} \cdot \ln(1 - p_i) \right]" />
        <p>
          where <Latex math="p_i" /> is the predicted frame win probability for match <Latex math="i" />,
          and <Latex math="s_{1,i}, s_{2,i}" /> are the actual frame scores.
          This treats each frame as a Bernoulli trial.
        </p>
        <p>Optimized using <code>scipy.optimize.minimize</code> (Nelder-Mead) over 34,521 matches.</p>

        <div className="grid-2" style={{ margin: '1rem 0' }}>
          <div className="stat">
            <div style={{ color: 'var(--green)', fontWeight: 700 }}>ELO</div>
            <div style={{ fontSize: '0.85rem', marginTop: '0.3rem' }}>K = 9.77, d = 327.15</div>
          </div>
          <div className="stat">
            <div style={{ color: 'var(--blue)', fontWeight: 700 }}>Glicko-2</div>
            <div style={{ fontSize: '0.85rem', marginTop: '0.3rem' }}><Latex math="\tau" /> = 1.488</div>
          </div>
        </div>
      </Section>

      <Section title="4. ML Models">
        <p>
          Three classifiers predict match outcomes using features from both rating systems:
        </p>
        <table>
          <thead>
            <tr><th>Model</th><th>Best Accuracy</th><th>Features</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Gradient Boosting</td>
              <td style={{ fontWeight: 700, color: 'var(--green)' }}>70.2%</td>
              <td>6 (ELO + Glicko-2 predictions + RD)</td>
            </tr>
            <tr><td>Logistic Regression</td><td>70.1%</td><td>35 (all features)</td></tr>
            <tr><td>Random Forest</td><td>69.5%</td><td>4 (Glicko-2 only)</td></tr>
            <tr><td>Pure ELO</td><td>68.4%</td><td>0 (rating threshold)</td></tr>
            <tr><td>Pure Glicko-2</td><td>68.0%</td><td>0 (rating threshold)</td></tr>
          </tbody>
        </table>
        <p style={{ marginTop: '1rem' }}>
          The two match win predictions (<Latex math="\text{elo\_match\_win\_rate}" /> and{' '}
          <Latex math="\text{glicko2\_match\_win\_rate}" />) account for <strong>74.5%</strong> of
          Gradient Boosting's feature importance. Adding 29 more features only improves accuracy by ~0.3%.
        </p>
      </Section>

      <Section title="5. Prediction Accuracy in Context">
        <table>
          <thead><tr><th>Sport</th><th>ELO / Rating</th><th>Best ML</th><th>Our Project</th></tr></thead>
          <tbody>
            <tr><td>Snooker</td><td>67-69%</td><td>69-71%</td><td style={{ fontWeight: 700, color: 'var(--green)' }}>70.2%</td></tr>
            <tr><td>Tennis</td><td>67-70%</td><td>69-75%</td><td>—</td></tr>
            <tr><td>Chess</td><td>65-69%</td><td>68-75%</td><td>—</td></tr>
            <tr><td>NBA</td><td>65-67%</td><td>67-70%</td><td>—</td></tr>
            <tr><td>Soccer (3-way)</td><td>51-55%</td><td>54-56%</td><td>—</td></tr>
          </tbody>
        </table>
        <p style={{ marginTop: '1rem' }}>
          The theoretical ceiling for pre-match snooker prediction is ~72-75%.
          The remaining gap is irreducible randomness: player form on the day,
          mental state, and match conditions that no pre-match model can capture.
        </p>
      </Section>
    </>
  )
}
