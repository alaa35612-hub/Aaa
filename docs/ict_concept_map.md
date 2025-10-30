# خريطة مفاهيم ICT في Smart Money Algo Pro E5

يوضح هذا المستند كيفية تمثيل مفاهيم ICT الرئيسية داخل المنفذ التنفيذي لمؤشر Smart Money Algo Pro E5 المكتوب بلغة Python. تم تجميع الروابط مباشرة من الكود لضمان التطابق بين الوظائف المرئية على الرسم البياني والمنطق البرمجي.

| المفهوم | التوصيف في الكود | مراجع السطور |
|---------|------------------|---------------|
| **BOS (Break of Structure)** | منطق تأكيد الكسر يحدث عندما تتحقق شروط _eval_condition في القسم البنيوي، مما يؤدي إلى ضبط الأعلام `isBosUp`/`isBosDn`، تحديث حالات `isCocUp`/`isCocDn`، واستدعاء `drawStructure("BOS", ...)`، مع استعمال النص `B O S` عند الرسم. كما يتم رسم العلامات عبر `drawLiveStrc` وإطلاق التنبيهات ذات الصلة. | `افضل واحد للان (1) (٣).py` ‎[L2114-L2122](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L2114-L2122), ‎[L6880-L6992](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L6880-L6992), ‎[L7118-L7120](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L7118-L7120) |
| **CHOCH (Change of Character)** | يتم ضبط الأعلام `isCocUp`/`isCocDn` عند تغير الاتجاه، مع استدعاء `drawStructure("ChoCh", ...)` عند التأكيد وإزالة التراكيب السابقة عبر `fixStrcAfterChoch`. يتم عرض النص `CHoCH` على الرسم باستخدام `drawLiveStrc`. | ‎[L2114-L2122](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L2114-L2122), ‎[L5874-L5882](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L5874-L5882), ‎[L6880-L6942](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L6880-L6942), ‎[L7118-L7119](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L7118-L7119) |
| **FVG (Fair Value Gap)** | يتم الاحتفاظ بالفجوات الصاعدة/الهابطة في مصفوفات مثل `bullish_gap_holder` و`bearish_gap_holder`. تقوم `_update_fvg` باكتشاف الفجوات حسب شروط الأسعار، وإنشاء الصناديق عبر `_fvg_create`، وتحديد الحالات `fvg_gap` و`fvg_removed` لإرسال التنبيهات «Break» أو «Found». | ‎[L2201-L2219](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L2201-L2219), ‎[L4895-L4997](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L4895-L4997) |
| **Order Block (OB)** | ترصد الدالة `ob_found` البلوكات الداخلية والخارجية وتعيد خصائص المنطقة ونوعها (-1 طلب شرائي، 1 عرض بيعي) مع نسب الحجم. تُخزن الحدود في مصفوفات مثل `ob_top`, `ob_btm`, `ob_type`، وتقوم `_filter_order_blocks` بتنظيفها، بينما يتكفل `_render_order_blocks` برسم الصناديق وإظهار بيانات الحجم والنسب. يتم تتبع حالات الكسر عبر الأعلام `bullish_OB_Break` و`bearish_OB_Break` مع إطلاق التنبيهات المناسبة. | ‎[L2185-L2219](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L2185-L2219), ‎[L4247-L4351](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L4247-L4351), ‎[L4414-L4642](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L4414-L4642) |
| **Liquidity Sweeps** | يتم استدعاء `sweepHL` عند فشل تأكيد BOS/CHOCH بعد اختراق قمة/قاع؛ حيث ترسم خطًا منقطًا وعلامة «X» عند تفعيل الخيار. يعتمد التوجيه على اتجاه `trend` لتحديد القمة أو القاع المستهدف. | ‎[L5874-L5909](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L5874-L5909), ‎[L6880-L6992](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L6880-L6992) |
| **OTE (Optimal Trade Entry)** | عند تفعيل `isOTE` يتم استخدام `drawPrevStrc` للحصول على مستويات فيبوناتشي المحددة بواسطة `ote1` (0.78) و`ote2` (0.61)، ثم رسم صندوق مظلل (`self.bxf`) مع تسمية «Golden zone» بين المستويين وتلوينه بلون `oteclr`. | ‎[L840-L841](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L840-L841), ‎[L7126-L7139](../%D8%A7%D9%81%D8%B6%D9%84%20%D9%88%D8%A7%D8%AD%D8%AF%20%D9%84%D9%84%D8%A7%D9%86%20(1)%20(%D9%A3).py#L7126-L7139) |

## قواعد الاستراتيجية (YAML)

يمثل الملف التالي توصيفًا بصيغة YAML لقواعد الاستراتيجية متعددة الأطر الزمنية المذكورة في نص المهمة:

```yaml
strategy:
  higher_timeframe: 15m
  lower_timeframe: 1m
  entry_rules:
    long_setup:
      HTF_trend: "Bullish BOS on 15m (uptrend structure confirmed)"
      LTF_conditions:
        - "Sell-side liquidity sweep on 1m (price sweeps a prior low then rebounds)"
        - "Retracement into a bullish FVG or demand OB within OTE zone (62%-79% retracement)"
    short_setup:
      HTF_trend: "Bearish BOS (or CHOCH) on 15m (downtrend structure)"
      LTF_conditions:
        - "Buy-side liquidity sweep on 1m (price sweeps a prior high then reverses)"
        - "Retracement into a bearish FVG or supply OB within OTE zone (62%-79% retracement)"
risk_management:
  risk_per_trade: 2%
  initial_account_size: $100
focus_sessions: ["London", "New York", "Asia"]
```
