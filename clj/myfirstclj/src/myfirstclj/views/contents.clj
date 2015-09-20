(ns
  ^{:author abc}
  myfirstclj.views.contents
  (:use [hiccup.form]
        [hiccup.element :only (link-to)]))

;https://github.com/yokolet/hiccup-samples
(defn index []
  [:div {:id "content"}
   [:h1 {:class "text-success"} "Abhishek with better "]])

(defn not-found []
  [:div
   [:h1 {:class "info-worning"} "Page Not Found"]
   [:p "There's no requested page. "]
   (link-to {:class "btn btn-primary"} "/" "Take me to Home")])

(def names
  ["John" "Mary" "Watson" "James"])

;http://codepen.io/nikhil/pen/GxhcD
;http://stackoverflow.com/questions/4712645/using-compojure-hiccup-and-ring-to-upload-a-file
(defn search-item
  []
  [:div {:class "well"}
   [:h1 {:class "text-info"} "Hello Hiccup and AngularJS"]
   [:div {:class "row"}
    [:div {:class "col-lg-2"}
     (label "search" "Search...:")]
    [:div {:class "col-lg-4"}
    [:form {:class "searchbox" :action ""}
    [:input {:type "search" :placeholder "search..."}
     [:ul {:class "suggestions"}
      (for [name names]
        [:li name])
      ]
     ]
     ]
     ]
     ]

   ]
  )

(defn hello []
  [:div {:class "well"}
   [:h1 {:class "text-info"} "Hello Hiccup and AngularJS"]
   [:div {:class "row"}
    [:div {:class "col-lg-2"}
     (label "name" "Name:")]
    [:div {:class "col-lg-6"}
     (text-field {:class "form-control" :ng-model "yourName" :placeholder "Enter a name here"} "your-name")]]
   [:hr]
   [:h1 {:class "text-success"} "Hello {{yourName}}!"]])