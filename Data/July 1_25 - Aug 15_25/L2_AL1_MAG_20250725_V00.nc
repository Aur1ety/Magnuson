<!DOCTYPE html>
<html class="login-pf">

<head>
    <meta charset="utf-8">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="robots" content="noindex, nofollow">

            <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>Sign in to Indian Space Science Data Center</title>
    <link rel="icon" href="/auth/resources/ida4f/login/keycloak/img/favicon.ico" />
            <link href="/auth/resources/ida4f/common/keycloak/web_modules/@patternfly/react-core/dist/styles/base.css" rel="stylesheet" />
            <link href="/auth/resources/ida4f/common/keycloak/web_modules/@patternfly/react-core/dist/styles/app.css" rel="stylesheet" />
            <link href="/auth/resources/ida4f/common/keycloak/node_modules/patternfly/dist/css/patternfly.min.css" rel="stylesheet" />
            <link href="/auth/resources/ida4f/common/keycloak/node_modules/patternfly/dist/css/patternfly-additions.min.css" rel="stylesheet" />
            <link href="/auth/resources/ida4f/common/keycloak/lib/pficon/pficon.css" rel="stylesheet" />
            <link href="/auth/resources/ida4f/login/keycloak/css/login.css" rel="stylesheet" />
            <link href="/auth/resources/ida4f/login/keycloak/css/tile.css" rel="stylesheet" />
</head>

<body class="">
<div class="login-pf-page">
    <div id="kc-header" class="login-pf-page-header">
        <div id="kc-header-wrapper"
             class="">Indian Space Science Data Center</div>
    </div>
    <div class="card-pf">
        <header class="login-pf-header">
                <h1 id="kc-page-title">        Sign in to your account

</h1>
      </header>
      <div id="kc-content">
        <div id="kc-content-wrapper">


    <div id="kc-form">
      <div id="kc-form-wrapper">
            <form id="kc-form-login" onsubmit="login.disabled = true; return true;" action="https://idp.issdc.gov.in/auth/realms/issdc/login-actions/authenticate?session_code=UXDAkMua7pYd0w79lKi4wNipEPi2pbQlNEXa1BaMfIA&amp;execution=4f75202b-ce79-477f-83e4-f05fbc4f5eb6&amp;client_id=al1-pradan&amp;tab_id=zEtwyk7hxuc" method="post">
                    <div class="form-group">
                        <label for="username" class="pf-c-form__label pf-c-form__label-text">Username or email</label>

                        <input tabindex="1" id="username" class="pf-c-form-control" name="username" value=""  type="text" autofocus autocomplete="off"
                               aria-invalid=""
                        />


                    </div>

                <div class="form-group">
                    <label for="password" class="pf-c-form__label pf-c-form__label-text">Password</label>

                    <input tabindex="2" id="password" class="pf-c-form-control" name="password" type="password" autocomplete="off"
                           aria-invalid=""
                    />


                </div>

                <div class="form-group login-pf-settings">
                    <div id="kc-form-options">
                        </div>
                        <div class="">
                                <span><a tabindex="5" href="/auth/realms/issdc/login-actions/reset-credentials?client_id=al1-pradan&amp;tab_id=zEtwyk7hxuc">Forgot Password?</a></span>
                        </div>

                  </div>

                  <div id="kc-form-buttons" class="form-group">
                      <input type="hidden" id="id-hidden-input" name="credentialId" />
                      <input tabindex="4" class="pf-c-button pf-m-primary pf-m-block btn-lg" name="login" id="kc-login" type="submit" value="Sign In"/>
                  </div>
            </form>
        </div>

    </div>





              <div id="kc-info" class="login-pf-signup">
                  <div id="kc-info-wrapper" class="">
            <div id="kc-registration-container">
                <div id="kc-registration">
                    <span>New user? <a tabindex="6"
                                                 href="/auth/realms/issdc/login-actions/registration?client_id=al1-pradan&amp;tab_id=zEtwyk7hxuc">Register</a></span>
                </div>
            </div>

                  </div>
              </div>
        </div>
      </div>

    </div>
  </div>
</body>
</html>
