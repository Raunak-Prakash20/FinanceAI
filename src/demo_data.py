"""Realistic sample SEC Form 10-Q filings for offline execution and testing."""

SAMPLE_NVDA_10Q_HTML = """
<!DOCTYPE html>
<html>
<head><title>NVIDIA CORP - Form 10-Q</title></head>
<body>
<h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
<h2>FORM 10-Q - QUARTERLY REPORT</h2>
<p>For the quarterly period ended October 29, 2023. Commission File Number: 000-23985</p>
<h3>NVIDIA CORPORATION</h3>

<div id="toc">
  <h2>Table of Contents</h2>
  <a href="#mda">Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</a>
  <a href="#risk">Item 1A. Risk Factors</a>
</div>

<hr />
<div id="mda">
<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>
Our revenue for the third quarter of fiscal year 2024 was $18.12 billion, up 206% from $5.93 billion a year ago, and up 34% sequentially.
The remarkable growth was driven by our Compute & Networking segment, particularly Data Center sales, which surged to a record $14.51 billion,
reflecting an increase of 279% year-over-year. Operating leverage was exceptionally strong as gross margin expanded to 74.0%, compared to 53.6%
in the prior year, primarily reflecting higher Data Center sales and favorable product mix. Operating income rose to $10.42 billion from $601 million
in the prior year, reflecting substantial operating leverage as operating expenses grew at a significantly slower rate than revenue.
Diluted EPS was $3.71, up over 12-fold from $0.27 in the prior year.
</p>
<p>
We are experiencing unprecedented global demand for accelerated computing and generative AI solutions. Our supply chain partners have expanded
capacity significantly, and we anticipate supply will continue to increase each quarter through next year. Forward guidance remains upbeat,
with revenue for the fourth quarter of fiscal 2024 expected to be $20.00 billion, plus or minus 2%, representing sustained top-line momentum.
Cash and cash equivalents and marketable securities were $18.28 billion, providing substantial liquidity.
</p>
</div>

<hr />
<div id="risk">
<h2>Item 1A. Risk Factors</h2>
<p>
Our business faces substantial risks related to customer concentration and export control regulations. A significant portion of our Data Center revenue
is concentrated among a limited number of cloud service providers and consumer internet companies. The loss of, or a significant reduction in purchases
by, any of these key customers could adversely affect our financial condition and results of operations.
</p>
<p>
Furthermore, recent US government export licensing regulations regarding advanced computing chips and semiconductor manufacturing items to China and
certain other countries may materially and adversely affect our business. During the third quarter of fiscal 2024, sales to China and other affected
destinations represented approximately 20-25% of our Data Center revenue. While demand in other regions is strong, we cannot assure that other geographies
will fully offset any long-term decline in China sales.
Additionally, rapid technological changes, supply chain disruptions for CoWoS packaging, and inventory build-up for older architecture components could
lead to inventory write-downs and margin contraction.
</p>
</div>

</body>
</html>
"""

SAMPLE_AAPL_10Q_HTML = """
<!DOCTYPE html>
<html>
<head><title>APPLE INC - Form 10-Q</title></head>
<body>
<h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
<h2>FORM 10-Q - QUARTERLY REPORT</h2>
<p>For the quarterly period ended December 30, 2023. Commission File Number: 001-36743</p>
<h3>APPLE INC.</h3>

<div id="mda">
<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>
Total net sales were $119.58 billion for the first quarter of fiscal 2024, up 2% from $117.15 billion in the prior-year quarter.
Products revenue was $96.46 billion, flat year-over-year, while Services revenue achieved an all-time record of $23.12 billion, up 11% year-over-year.
Gross margin for the quarter expanded to 45.9%, up 290 basis points from 43.0% in the prior-year period, driven by favorable mix toward higher-margin
Services and cost savings across hardware components. Operating income rose to $40.37 billion compared to $36.02 billion a year ago.
Diluted EPS was $2.18, up 16% year-over-year, reflecting operating leverage and ongoing share repurchases.
</p>
<p>
Our active installed base of devices has now surpassed 2.2 billion active devices, reaching an all-time high across all products and geographic segments.
Operating cash flow generated during the quarter was $39.9 billion, allowing us to return nearly $27 billion to shareholders through dividends
and share repurchases. Management remains optimistic about long-term ecosystem monetization and expansion in emerging markets.
</p>
</div>

<div id="risk">
<h2>Item 1A. Risk Factors</h2>
<p>
Global macroeconomic volatility, foreign exchange headwinds, and intense smartphone competition in Greater China present ongoing risks.
Net sales in Greater China declined 13% during the quarter due to competitive pressure and consumer spending caution.
Continued weakness in key international markets could depress hardware upgrades and pressure average selling prices.
</p>
<p>
We also face intensifying regulatory scrutiny globally regarding the App Store, digital marketplace competition, and platform interoperability,
notably under the European Union Digital Markets Act (DMA). Changes required to comply with the DMA could impact developer fees, reduce Services
operating margins, and increase compliance overhead. Furthermore, high reliance on single-source suppliers for advanced silicon and assembly
in Asia leaves our manufacturing operations exposed to regional geopolitical friction and logistical bottlenecks.
</p>
</div>

</body>
</html>
"""

SAMPLE_MSFT_10Q_HTML = """
<!DOCTYPE html>
<html>
<head><title>MICROSOFT CORP - Form 10-Q</title></head>
<body>
<h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
<h2>FORM 10-Q - QUARTERLY REPORT</h2>
<p>For the quarterly period ended December 31, 2023. Commission File Number: 001-14278</p>
<h3>MICROSOFT CORPORATION</h3>

<div id="mda">
<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>
Revenue was $62.02 billion for the second quarter of fiscal 2024, an increase of 18% compared to $52.75 billion in the prior year.
Operating income was $27.03 billion, up 33% year-over-year, reflecting strong operational execution and operating leverage.
Diluted EPS reached $2.93, increasing 33% from the prior year.
Growth was led by Microsoft Cloud, which generated $33.7 billion in revenue, up 24% year-over-year. Azure and other cloud services revenue
grew 30%, with 6 points of growth attributable to artificial intelligence services.
Commercial bookings grew 17% driven by large, long-term Azure commitments.
</p>
<p>
Operating expenses increased 3% to $14.4 billion, demonstrating disciplined cost management while accelerating capital expenditures for AI infrastructure.
Free cash flow was $9.1 billion, supporting ongoing investments in cloud data center capacity. Forward guidance points to continued double-digit cloud growth.
</p>
</div>

<div id="risk">
<h2>Item 1A. Risk Factors</h2>
<p>
Rapidly scaling cloud and AI infrastructure requires significant capital expenditure, which could pressure free cash flow and near-term operating margins
if capacity utilization lags capital deployment. Capital expenditures including finance leases were $11.5 billion during the quarter.
</p>
<p>
Additionally, cybersecurity threats, potential service outages in Azure data centers, and third-party dependency on power and specialized GPU hardware
pose operational vulnerabilities. The integration of Activision Blizzard introduces operational and regulatory compliance complexities across multiple
jurisdictions. Increasing competition in enterprise AI and productivity software could impact pricing power and renewal rates.
</div>

</body>
</html>
"""

SAMPLE_NVDA_PRIOR_10Q_HTML = """
<!DOCTYPE html>
<html>
<head><title>NVIDIA CORP - Form 10-Q Prior</title></head>
<body>
<h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
<h2>FORM 10-Q - QUARTERLY REPORT</h2>
<p>For the quarterly period ended July 30, 2023. Commission File Number: 000-23985</p>
<h3>NVIDIA CORPORATION</h3>

<div id="mda">
<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>
Our revenue for the second quarter of fiscal year 2024 was $13.51 billion, up 101% from $6.70 billion a year ago, and up 88% sequentially.
The remarkable growth was driven by our Compute & Networking segment, particularly Data Center sales, which surged to a record $10.32 billion,
reflecting an increase of 171% year-over-year. Operating leverage was exceptionally strong as gross margin expanded to 70.1%, compared to 43.5%
in the prior year, primarily reflecting higher Data Center sales and favorable product mix. Operating income rose to $6.80 billion from $499 million
in the prior year, reflecting substantial operating leverage as operating expenses grew at a significantly slower rate than revenue.
Diluted EPS was $2.48, up over 9-fold from $0.26 in the prior year.
</p>
<p>
We are experiencing significant global demand for accelerated computing architectures across cloud hyperscalers. Our supply chain partners
have expanded capacity significantly, and we anticipate supply will continue to increase each quarter through next year. Forward guidance remains upbeat,
with revenue for the third quarter of fiscal 2024 expected to be $16.00 billion, plus or minus 2%, representing sustained top-line momentum.
Cash and cash equivalents and marketable securities were $16.02 billion, providing substantial liquidity.
</p>
</div>

<div id="risk">
<h2>Item 1A. Risk Factors</h2>
<p>
Our business faces substantial risks related to customer concentration and export control regulations. A significant portion of our Data Center revenue
is concentrated among a limited number of cloud service providers and consumer internet companies. The loss of, or a significant reduction in purchases
by, any of these key customers could adversely affect our financial condition and results of operations.
</p>
<p>
Global macroeconomic conditions and geopolitical developments, including potential changes to export regulations, could affect our shipments to certain
international markets. Supply constraints for specialized packaging and raw materials may also impact our ability to meet customer delivery schedules.
Additionally, rapid technological changes, supply chain disruptions for CoWoS packaging, and inventory build-up for older architecture components could
lead to inventory write-downs and margin contraction.
</p>
</div>

</body>
</html>
"""

SAMPLE_AAPL_PRIOR_10Q_HTML = """
<!DOCTYPE html>
<html>
<head><title>APPLE INC - Form 10-Q Prior</title></head>
<body>
<h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
<h2>FORM 10-Q - QUARTERLY REPORT</h2>
<p>For the quarterly period ended September 30, 2023. Commission File Number: 001-36743</p>
<h3>APPLE INC.</h3>

<div id="mda">
<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>
Total net sales were $89.50 billion for the fourth quarter of fiscal 2023, down 1% from $90.15 billion in the prior-year quarter.
Products revenue was $67.18 billion, while Services revenue achieved an all-time record of $22.31 billion, up 16% year-over-year.
Gross margin for the quarter was 45.2%, compared to 42.3% in the prior-year period. Operating income was $26.97 billion compared to $27.02 billion a year ago.
Diluted EPS was $1.46, up 13% year-over-year, reflecting operating leverage and ongoing share repurchases.
</p>
<p>
Our active installed base of devices has now surpassed 2.0 billion active devices, reaching an all-time high across all products and geographic segments.
Operating cash flow generated during the quarter was $21.6 billion, allowing us to return nearly $25 billion to shareholders through dividends
and share repurchases. Management remains optimistic about long-term ecosystem monetization and expansion in emerging markets.
</p>
</div>

<div id="risk">
<h2>Item 1A. Risk Factors</h2>
<p>
Global macroeconomic volatility, foreign exchange headwinds, and intense smartphone competition in Greater China present ongoing risks.
Net sales in Greater China declined during the quarter due to competitive pressure and consumer spending caution.
Continued weakness in key international markets could depress hardware upgrades and pressure average selling prices.
</p>
<p>
We also face intensifying regulatory scrutiny globally regarding the App Store, digital marketplace competition, and platform interoperability.
Furthermore, high reliance on single-source suppliers for advanced silicon and assembly in Asia leaves our manufacturing operations exposed to
regional geopolitical friction and logistical bottlenecks.
</p>
</div>

</body>
</html>
"""

SAMPLE_MSFT_PRIOR_10Q_HTML = """
<!DOCTYPE html>
<html>
<head><title>MICROSOFT CORP - Form 10-Q Prior</title></head>
<body>
<h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
<h2>FORM 10-Q - QUARTERLY REPORT</h2>
<p>For the quarterly period ended September 30, 2023. Commission File Number: 001-14278</p>
<h3>MICROSOFT CORPORATION</h3>

<div id="mda">
<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>
Revenue was $56.52 billion for the first quarter of fiscal 2024, an increase of 13% compared to $50.12 billion in the prior year.
Operating income was $24.16 billion, up 25% year-over-year, reflecting strong operational execution and operating leverage.
Diluted EPS reached $2.99, increasing 27% from the prior year.
Growth was led by Microsoft Cloud, which generated $31.8 billion in revenue, up 24% year-over-year. Azure and other cloud services revenue
grew 29%, with 3 points of growth attributable to artificial intelligence services.
</p>
<p>
Operating expenses increased 1% to $13.3 billion, demonstrating disciplined cost management while accelerating capital expenditures for AI infrastructure.
Free cash flow was $20.7 billion, supporting ongoing investments in cloud data center capacity. Forward guidance points to continued double-digit cloud growth.
</p>
</div>

<div id="risk">
<h2>Item 1A. Risk Factors</h2>
<p>
Rapidly scaling cloud and AI infrastructure requires significant capital expenditure, which could pressure free cash flow and near-term operating margins
if capacity utilization lags capital deployment. Capital expenditures including finance leases were $9.9 billion during the quarter.
</p>
<p>
Additionally, cybersecurity threats, potential service outages in Azure data centers, and third-party dependency on power and specialized GPU hardware
pose operational vulnerabilities. The integration of Activision Blizzard introduces operational and regulatory compliance complexities across multiple
jurisdictions. Increasing competition in enterprise AI and productivity software could impact pricing power and renewal rates.
</p>
</div>

</body>
</html>
"""

