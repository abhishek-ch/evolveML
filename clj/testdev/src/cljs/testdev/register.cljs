 (ns testdev.register)

;http://yogthos.github.io/reagent-forms-example.html

(defn friend-source [text]
 (filter
  #(-> % (.toLowerCase %) (.indexOf text) (> -1))
  ["Alice" "Alan" "Bob" "Beth" "Jim" "Jane" "Kim" "Rob" "Zoe"]))

(defn test []
      [:div {:class "col-sm-8 col-sm-offset-2 text-center"}
       [:h2 "Debdoot Agency Group"]
       [:hr]
       [:h4 "Distributor & Wholeseller"]
       [:p
        "Connect or Contact us for knowing about our clients and Nature of Business"
        ]
       ])

(defn test_register []
[:div {:field :typeahead
       :id :ta
       :input-placeholder "pick a friend"
       :data-source friend-source
       :input-class "form-control"
       :list-class "typeahead-list"
       :item-class "typeahead-item"
       :highlight-class "highlighted"}])