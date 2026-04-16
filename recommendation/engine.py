# engine.py

import os
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Product, ProductVariation

class MakeupRecommender:
    """
    Enhanced hybrid recommender:
      1. Builds a TF-IDF index over Product.name + Product.detailed_description + Product.tags.
      2. Computes rule-based scores (hard exclusions + boosts).
      3. Optionally incorporates Product.popularity_score.
      4. Enforces shade-level matching via ProductVariation.
    """

    def __init__(self, content_weight=0.4, rule_weight=0.4, popularity_weight=0.2):
        self.w_content = content_weight
        self.w_rule = rule_weight
        self.w_popular = popularity_weight

        # === 1) Build TF-IDF index ===
        #   We load all in-stock products, concatenate their name/description/tags into a single string,
        #   then vectorize once so that _content_based(...) can compute cosine similarities quickly.
        #
        # REQUIREMENTS:
        #   - Product model must have: id, name, detailed_description (TextField), tags (TextField).
        #
        queryset = Product.objects.filter(quantity__gt=0).only(
            "id", "name", "detailed_description", "tags"
        )

        corpus_texts = []
        self.prod_ids = []
        for p in queryset:
            # Combine name + description + tags into one “document” per product
            combined = " ".join(
                filter(
                    None,
                    [
                        p.name or "",
                        p.detailed_description or "",
                        p.tags or "",
                    ],
                )
            ).lower()
            corpus_texts.append(combined)
            self.prod_ids.append(p.id)

        # If there are no products, avoid errors
        if corpus_texts:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.prod_tfidf = self.vectorizer.fit_transform(corpus_texts)
        else:
            self.vectorizer = None
            self.prod_tfidf = None

    def _get_popular_fallback(self, top_n=5):
        products = Product.objects.filter(quantity__gt=0).order_by('-popularity_score')[:top_n]
        recs = []
        for p in products:
            recs.append({
                "product": p,
                "combined_score": 0.3, 
                "reason": "Popular choice",
            })
        return recs

    def get_recommendations(self, customer, top_n=10):

        # 1) Retrieve the latest SkinAssessment for this customer (or a TemporaryAssessment)
        try:
            assess = customer.skin_assessments.first()
        except:
            from .models import TemporaryAssessment
            assess = TemporaryAssessment()  # fallback if none exists

        # 2) Compute per-product scores
        content_scores = self._content_based(assess)
        rule_scores = self._rule_based(assess)
        pop_scores = self._popularity_based()

        # 3) Normalize each dictionary to [0,1]
        all_ids = set(content_scores.keys()) | set(rule_scores.keys()) | set(pop_scores.keys())
        c_norm = self._normalize_dict(content_scores, all_ids)
        r_norm = self._normalize_dict(rule_scores, all_ids)
        p_norm = self._normalize_dict(pop_scores, all_ids)

        # 4) Blend them into a final score
        final_scores = {}
        for pid in all_ids:
            score = (
                self.w_content * c_norm.get(pid, 0.0)
                + self.w_rule * r_norm.get(pid, 0.0)
                + self.w_popular * p_norm.get(pid, 0.0)
            )
            final_scores[pid] = score

            if not final_scores:
                return self._get_popular_fallback()

        # 5) Sort product IDs by descending final score
        sorted_pairs = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_pairs = sorted_pairs[:top_n]
        top_pids = [pid for pid, _ in top_pairs]

        # 6) Fetch actual Product instances
        prods = Product.objects.filter(id__in=top_pids).only(
            "id",
            "name",
            "price",
            "popularity_score",
            "finish",
            "texture",
            "is_hypoallergenic",
            "skin_condition",
            "spf",
        )
        prod_map = {p.id: p for p in prods}

        # 7) Build return list with “reason” strings
        recommendations = []
        for pid, combined_score in top_pairs:
            prod = prod_map.get(pid)
            if not prod:
                continue

            reasons = []
            # a) Content‐based snippet
            c_val = c_norm.get(pid, 0.0)
            if c_val > 0:
                reasons.append(f"Content similarity: {round(c_val, 2)}")

            # b) Rule‐based snippet
            r_val = r_norm.get(pid, 0.0)
            if r_val > 0:
                reasons.append(f"Rule boost: {round(r_val, 2)}")

            # c) Shade‐matching snippet
            #    If any variation matches the customer's undertone+surface_tone exactly
            var_matches = ProductVariation.objects.filter(
                product_id=pid,
                skin_tone=assess.undertone,
                surface_tones=customer.surface_tone,
            ).exists()
            if var_matches:
                reasons.append("Exact shade available")

            # d) Popularity snippet (optional to show)
            p_val = p_norm.get(pid, 0.0)
            if p_val > 0:
                reasons.append(f"Popularity: {round(p_val, 2)}")

            reason_str = " | ".join(reasons) if reasons else "No strong signals"

            recommendations.append(
                {
                    "product": prod,
                    "combined_score": round(combined_score, 4),
                    "reason": reason_str,
                }
            )

        return recommendations

    def _content_based(self, assess):
        """
        Returns a dict {product_id: similarity_score}. Uses TF-IDF on (name + description + tags).
        The “user document” is built from assess.skin_type, assess.finish_preference, assess.texture_preference,
        and all comma‐separated skin concerns (e.g. “acne,redness”).
        """
        scores = {}
        if not self.vectorizer or self.prod_tfidf is None:
            return scores

        # Build a “user text” string
        user_tokens = [
            assess.skin_type or "",
            assess.finish_preference or "",
            assess.texture_preference or "",
            assess.surface_tone or "",
            assess.undertone or "",
        ]
        if assess.concerns:
            # assume assess.concerns is a comma‐separated string
            user_tokens += assess.concerns.split(",")
        user_text = " ".join(token.strip() for token in user_tokens if token).lower()

        if not user_text.strip():
            return scores  # no basis for content similarity

        user_vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, self.prod_tfidf).flatten()

        # Only keep those with positive similarity
        for idx, sim_val in enumerate(sims):
            if sim_val > 0:
                pid = self.prod_ids[idx]
                # Before accepting, ensure at least one ProductVariation is compatible:
                # (either exact shade match or we leave that for rule‐based logic to penalize)
                scores[pid] = float(sim_val)

        return scores

    def _rule_based(self, assess):
        """
        Returns a dict {product_id:
        rule_score}. Applies:
          - Hard exclusion for high sensitivity if product isn't hypoallergenic
          - Boosts for dryness/oiliness/aging/acne rules
          - Shade‐availability downweight: if no matching variation, assign 0 entirely
        """
        scores = {}

        # Preload all in-stock products (only fields needed)
        queryset = Product.objects.filter(quantity__gt=0).only(
            "id",
            "finish",
            "texture",
            "is_hypoallergenic",
            "skin_condition",
            "spf",
        )

        for prod in queryset:
            pid = prod.id
            rule_score = 0.0

            # 1) HIGH SENSITIVITY → must be hypoallergenic, otherwise exclude
            if assess.sensitivity_level is not None and assess.sensitivity_level >= 4:
                if not prod.is_hypoallergenic:
                    continue
                else:
                    rule_score += 0.6

            # 2) OILY SKIN → prefer matte/powder
            if assess.oiliness_level is not None and assess.oiliness_level >= 4:
                if prod.finish == "matte":
                    rule_score += 0.5
                if prod.texture == "powder":
                    rule_score += 0.3

            # 3) DRY SKIN → prefer dewy/cream/liquid
            if assess.hydration_level is not None and assess.hydration_level <= 2:
                if prod.finish == "dewy":
                    rule_score += 0.5
                if prod.texture in ["cream", "liquid"]:
                    rule_score += 0.3

            # 4) ACNE PRONE → exclude products whose tags include oily ingredients
            if assess.acne_proneness is not None and assess.acne_proneness >= 4:
                bad_tokens = {"oil", "mineral_oil", "lanolin"}
                prod_tags = {
                    t.strip().lower() for t in (prod.skin_condition or "").split(",")
                }
                if bad_tokens & prod_tags:
                    continue
                if "non_comedogenic" in prod_tags:
                    rule_score += 0.4

            # 5) AGING CONCERNS → boost “anti_aging” or “hydrating”
            if assess.aging_concerns is not None and assess.aging_concerns >= 3:
                prod_tags = {
                    t.strip().lower() for t in (prod.skin_condition or "").split(",")
                }
                if "anti_aging" in prod_tags or "hydrating" in prod_tags:
                    rule_score += 0.4

            # 6) SPF BONUS (small)
            if prod.spf and prod.spf >= 15:
                rule_score += 0.1

            # 7) SHADE AVAILABILITY CHECK (hard): if no ProductVariation matches undertone+surface_tone, exclude
            shade_ok = ProductVariation.objects.filter(
                product_id=pid,
                skin_tone=assess.undertone,
                surface_tones=assess.surface_tone,
            ).exists()
            if not shade_ok:
                rule_score -= 0.3

            if rule_score > 0:
                scores[pid] = rule_score

        return scores

    def _popularity_based(self):
        """
        Returns {product_id: normalized_popularity_score}. Normalizes Product.popularity_score across in-stock products.
        """
        scores = {}
        queryset = Product.objects.filter(quantity__gt=0).only("id", "popularity_score")
        max_pop = 0.0
        for p in queryset:
            if p.popularity_score is not None and p.popularity_score > max_pop:
                max_pop = p.popularity_score

        if max_pop <= 0:
            # everything gets 0 if there's no popularity data
            return {p.id: 0.0 for p in queryset}

        for p in queryset:
            norm = (p.popularity_score or 0.0) / max_pop
            scores[p.id] = norm

        return scores

    def _normalize_dict(self, score_dict, all_ids):
        """
        Given a dict {id: raw_score} and the full set of IDs, returns {id: normalized_score}
        where normalized_score = raw_score / max_value_in_dict. If max ≤ 0, everyone → 0.0.
        Missing IDs are assigned 0.0.
        """
        if not score_dict:
            return {pid: 0.0 for pid in all_ids}
        max_val = max(score_dict.values())
        if max_val <= 0:
            return {pid: 0.0 for pid in all_ids}
        return {pid: (score_dict.get(pid, 0.0) / max_val) for pid in all_ids}
