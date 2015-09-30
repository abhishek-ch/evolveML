(ns testdev.about)





(def click-count (reagent.core/atom 0))

(defn counting-component []
      [:div
       "The atom " [:code "click-count"] " has value: "
       @click-count ". "
       [:input {:type "button" :value "Click me!"
                :on-click #(swap! click-count inc)}]])


(defn timer-component []
      (let [seconds-elapsed (reagent.core/atom 1000)]
           (fn []
               (js/setTimeout #(swap! seconds-elapsed inc) 1000)
               [:div
                "Seconds Elapsed: " @seconds-elapsed])))


(defn atom-input [value]
      [:input {:type "text"
               :value @value
               :on-change #(reset! value (-> % .-target .-value))}])

(defn shared-state []
      (let [val (reagent.core/atom "foo")]
           (fn []
               [:div
                [:p "The value is now: " @val]
                [:p "Change it here: " [atom-input val]]])))



