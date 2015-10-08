(ns testdev.core
  (:require [reagent.core :as reagent :refer [atom]]
            [reagent.session :as session]
            [secretary.core :as secretary :include-macros true]
            [goog.events :as events]
            [goog.history.EventType :as EventType]
            [testdev.contents :as contents]
            [testdev.about :as aboutpage]
            )
  (:import goog.History))

;http://www.mattgreer.org/articles/reagent-rocks/
;http://stackoverflow.com/questions/31009978/on-click-handler-for-a-list-item-reagent-clojurescript
;http://getbootstrap.com/components/

(defn loadHome []

  [:header {:class "masthead"}
   [:div {:class "container"}
    [:div {:class "row"}
     [:div {:class "col-sm-6"}
      [:h1
       [:a {:title "Abhishek Testing"} "Debdoot Agency##..."]
       [:p {:class "lead"} "{A-z of pharmaceutical need}"]
       ]]

     [:div {:class "col-sm-6"}
      [:div {:class "pull-right  hidden-xs"}
       [:a {:href "#" :class "dropdown-toggle" :data-toggle "dropdown"}
        [:h3
         [:i {:class "glyphicon glyphicon-cog"}]]]
       [:ul {:class "dropdown-menu"}
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-chevron-right"}]
          "Link1"]]
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-user"}]
          "Link2"]]
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-lock"}]
          "Link3"]]
        [:li
         [:a {:href "http://www.google.com"}
          [:i {:class "glyphicon glyphicon-cog"}]
          "Link4"]]
        ]]]

     ]]]

  )

;Main page scrollable
(defn page [body]
  [:div.page
   [:div {:id "wrap"}
    body
    [:div {:class "divider" :id "section0"}]
    [:nav {:class "navbar navbar-custom navbar-inverse navbar-static-top" :id "nav" :data-spy "affix" :data-offset-top "100"} (contents/fixed-navigation-bar)]
    [:div {:class "divider" :id "section1"}]
    [:div {:class "container"} (contents/pagecontents)]
    [:div {:class "divider" :id "section2"}]
    [:section {:class "bg-1"} (contents/pageWithImageOne)]
    [:div {:class "divider" :id "section3"}]
    [:section {:class "container" :id "weare"} (contents/pageOwn)]
    [:div {:class "divider"}]
    [:section {:class "bg-5" :id "section4"} (contents/explainDelivery)]
    [:div {:class "container bg-4"} [:div {:class "row"}(contents/delegatePictureBlock)]]
    [:section {:class "divider" :id "section5"}]
    [:div {:class "row"} (contents/direction)]
    [:div {:class "container"} (contents/beforefooter)]
    [:div {:id "footer"} (contents/footer)]
    [:ul {:class "nav pull-right scroll-top"} (contents/pull-up)]
    ]]
  )



(defn check-nil-then-predicate
      "Check if the value is nil, then apply the predicate"
      [value predicate]
      (if (nil? value)
        false
        (predicate value)))


(defn valid-email?
      [email]
      (check-nil-then-predicate email (fn [arg] (boolean (first (re-seq #"\w+@\w+.\w+" arg))))))


(defn eight-or-more-characters?
      [word]
      (check-nil-then-predicate word (fn [arg] (> (count arg) 7))))


(defn has-special-character?
      [word]
      (check-nil-then-predicate word (fn [arg] (boolean (first (re-seq #"\W+" arg))))))


(defn has-number?
      [word]
      (check-nil-then-predicate word (fn [arg] (boolean (re-seq #"\d+" arg)))))


(defn prompt-message
      "A prompt that will animate to help the user with a given input"
      [message]
      [:div {:class "my-messages"}
       [:div {:class "prompt message-animation"} [:p message]]])


(defn input-element
      "An input element which updates its value and on focus parameters on change, blur, and focus"
      [id name type value in-focus]
      [:input {:id id
               :name name
               :class "form-control"
               :type type
               :required ""
               :value @value
               :on-change #(reset! value (-> % .-target .-value))
               :on-focus #(swap! in-focus not)
               :on-blur (fn [arg] (if (nil? @value) (reset! value ""))(swap! in-focus not))}])


(defn input-and-prompt
      "Creates an input box and a prompt box that appears above the input when the input comes into focus. Also throws in a little required message"
      [label-value input-name input-type input-element-arg prompt-element required?]
      (let [input-focus (atom false)]
           (fn []
               [:div
                [:label label-value]
                (if @input-focus prompt-element [:div])
                [input-element input-name input-name input-type input-element-arg input-focus]
                (if (and required? (= "" @input-element-arg))
                  [:div "Field is required!"]
                  [:div])])))


(defn email-form [email-address-atom]
      (input-and-prompt "email"
                        "email"
                        "email"
                        email-address-atom
                        (prompt-message "What's your email address?")
                        true))


(defn name-form [name-atom]
      (input-and-prompt "name"
                        "name"
                        "text"
                        name-atom
                        (prompt-message "What's your name?")
                        true))


(defn password-requirements
      "A list to describe which password requirements have been met so far"
      [password requirements]
      [:div
       [:ul (->> requirements
                 (filter (fn [req] (not ((:check-fn req) @password))))
                 (map (fn [req] ^{:key req} [:li (:message req)])))]])


(defn password-form
      [password]
      (let [password-type-atom (atom "password")]
           (fn []
               [:div
                [(input-and-prompt "password"
                                   "password"
                                   @password-type-atom
                                   password
                                   (prompt-message "What's your password")
                                   true)]
                [password-requirements password [{:message "8 or more characters" :check-fn eight-or-more-characters?}
                                                 {:message "At least one special character" :check-fn has-special-character?}
                                                 {:message "At least one number" :check-fn has-number?}]]])))


(defn wrap-as-element-in-form
      [element]
      [:div {:class="row form-group"}
       element])


;;refered from https://github.com/jkk/formative
(defn about-page1 []

      [:div
       [:div [:h2 "About testdev"]
        [:div [:a {:href "#/"} "go to the home page"]]
        [:div {:class "container"} (aboutpage/pickabout)]
        ]
       [:div {:class "container"} (aboutpage/home-page-abt)]
       ]
      )

;;call with # as prefix /#/about
(defn about-page []
      (let [email-address (atom nil)
            name (atom nil)
            password (atom nil)]
           (fn []
               [:div {:class "signup-wrapper"}
                [:h2 "Welcome to DhruvChimp"]
                [:form
                 (wrap-as-element-in-form [email-form email-address])
                 (wrap-as-element-in-form [name-form name])
                 (wrap-as-element-in-form [password-form password])]])))


;;Home Page
(defn home-page []
      (reagent/render [page [loadHome]]
                      (.-body js/document)))

;;Home Page
(defn home-page-abt []
  (reagent/render [page [loadHome]]
                  (.-body js/document)))


;; -------------------------
;; Views




(defn current-page []
  [:div [(session/get :current-page)]])


(defonce selected-department (atom "department!"))

(defn simple-component []
  [:div#sidebar-wrapper
   [:ul.sidebar-nav
    [:li.sidebar-brand [:a {:href "#"} "Departments"]]
    [:li [:a {:on-click #(reset! selected-department "Dairy") :href "#"} "Dairy"]]
    [:li [:a {:on-click #(reset! selected-department "Deli") :href "#"} "Deli"]]
    [:li [:a {:on-click #(reset! selected-department "Grocery") :href "#"} "Grocery"]]]
   [:label @selected-department]])


(defn render-simple []
  (reagent/render-component [simple-component]
                            (.-body js/document)))

;; -------------------------
;; Routes
(secretary/set-config! :prefix "#")

(secretary/defroute "/" []
                    (session/put! :current-page #'home-page))

(secretary/defroute "/about" []
                    (session/put! :current-page #'about-page))

(secretary/defroute "/render" []
                    (session/put! :current-page #'render-simple))

;; -------------------------
;; History
;; must be called after routes have been defined
(defn hook-browser-navigation! []
  (doto (History.)
    (events/listen
      EventType/NAVIGATE
      (fn [event]
        (secretary/dispatch! (.-token event))))
    (.setEnabled true)))

;; -------------------------
;; Initialize app
(defn mount-root []
  (reagent/render [current-page] (.getElementById js/document "app")))

(defn init! []
  (hook-browser-navigation!)
  (mount-root))
